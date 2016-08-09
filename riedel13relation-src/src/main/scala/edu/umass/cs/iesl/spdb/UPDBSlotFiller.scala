package edu.umass.cs.iesl.spdb

import cc.factorie.app.nlp._
import cc.factorie.db.mongo.{LazyCubbieConverter, MongoCubbieCollection}
import com.mongodb.{Mongo, DBCollection}
import com.typesafe.config.{Config, ConfigFactory}
import parse.ParseTree
import xml.XML
import io.Source
import cc.factorie.app.strings.Stopwords
import org.bson.types.ObjectId
import collection.mutable.{HashSet, ArrayBuffer, HashMap}
import java.util.Calendar
import java.nio.channels.Channels
import java.io.{FileOutputStream, File, PrintStream}

/**
 * @author sriedel
 */
object UPDBSlotFiller extends HasLogger {

  type DocCubbie = DocumentCubbie[TokenCubbie with TokenPOSCubbie with TokenNerCubbie with TokenLemmaCubbie, SentenceCubbie with SentenceParseCubbie, TokenSpanCubbie]

  val targetSlots = Set("per:member_of", "per:employee_of", "org:top_members/employees")

  def extractMentionPairFeaturesAndAddCandidateSlots(spdb: SelfPredictingDatabase, documents: Iterator[DocCubbie],
                                                     mentions: MongoCubbieCollection[Mention],
                                                     linker: (Document, Mention) => Entity,
                                                     queryEntity: Entity => Boolean,
                                                     goldSlot: (Entity, Entity, String) => Boolean) {
    for (cd <- documents) {
      val doc = cd.fetchDocument
      //      println("#Document " + doc.name)
      val mentionSeq = mentions.query(_.docId(doc.name)).toBuffer
      val mention2Entity = mentionSeq.map(m => m -> linker(doc, m)).toMap

      for (source <- mentionSeq; dest <- mentionSeq; if (doc.tokens(source.tokenBegin()).sentence.start == doc.tokens(dest.tokenBegin()).sentence.start //the same sentence
        && source.phrase() != dest.phrase())) {

        val arg1 = mention2Entity(source)
        val arg2 = mention2Entity(dest)

        val tuple = IndexedSeq(arg1.id, arg2.id)

        if (arg1.tacId.opt.isDefined) {

          val isDevQuery = queryEntity(arg1)

          //          println("Arg1: " + arg1)
          //          println("Arg2: " + arg2)

          val features = extractFeatures(source, dest, doc)

          for (feat <- features) {
            spdb.addCell(feat, tuple, 1.0, false, false, true)
          }

          //add candidate slots
          //todo: we should probably incorporate some filtering heuristics here.
          for (slot <- targetSlots) {
            val cell = spdb.getCell(slot, tuple)
            if (!cell.isDefined) {
              val goldValue = if (goldSlot(arg1, arg2, slot)) 1.0 else 0.0
              spdb.addCell(slot, tuple, goldValue, isDevQuery, false, true)
            }
          }

          //          println(features.mkString("\t"))
        }

      }
    }
  }

  def extractFeatures(source: Mention, dest: Mention, doc: Document): ArrayBuffer[String] = {
    val features = new ArrayBuffer[String]
    val sentence = doc.tokens(source.tokenBegin()).sentence
    val shead = EntityMentionUtils.headOfMention(source, sentence)
    val dhead = EntityMentionUtils.headOfMention(dest, sentence)
    val (rootOfPath, pathString, path) = DepUtils.find_path(shead, dhead, sentence.attr[ParseTree])
    if (pathString != "junk" && pathString != "exception" && path.size < 10) {
      val pathContentWords = path.drop(1).dropRight(1).filter(_._1.attr[NerTag].value == "O").map(_._1.attr[Lemma].value.toLowerCase).filter(!Stopwords.contains(_)) // only none-ner tokens can be trigger words

      //  val trigger = if (pathContentWords.size == 0) rootOfPath else pathContentWords.mkString(",")

      // features += "trigger#" + trigger
      //features += "source#" +  EntityMentionUtils.wordPOS(source,sentence)
      //features += "dest#" + EntityMentionUtils.wordPOS(dest,sentence)
      if (pathString.length() > 0)
        features += "path#" + pathString

      //between context, newly added
      if (dest.tokenBegin() - source.tokenEnd() < 20) {
        val lexContext = EntityMentionUtils.betweenLexContext(source, dest, sentence)
        val posContext = EntityMentionUtils.betweenPOSContext(source, dest, sentence)
        if (lexContext.filter(_.length() > 0).size > 0) {
          features += lexContext.filter(_.length() > 0).map(x => "lex#" + x).mkString("\t")
          features += posContext.filter(_.length() > 0).map(x => "pos#" + x).mkString("\t")
        }
      }

      //lc, rc, words to the left of the source arg, to the right of the dest arg
      val neighborContext = EntityMentionUtils.neighbor(source, dest, sentence)
      if (neighborContext.length() > 0) {
        val contexts = neighborContext.split("\t")
        features ++= contexts
      }
    }
    features.filter(_.length() > 0).filter(x => !(x.contains("javascript")))
  }


  def main(args: Array[String]) {

    Conf.add(args(0))
    import Conf._

    val mongoConn = new Mongo(conf.getString("source-data.mongo.host"), conf.getInt("source-data.mongo.port"))
    val mongoDB = mongoConn.getDB(conf.getString("source-data.mongo.db")) // web data for tac-kbp
    val colldoc = mongoDB.getCollection("documents")
    val collmention = mongoDB.getCollection("mentions")

    val documents = new MongoCubbieCollection(colldoc,
      () => new DocumentCubbie(() => new TokenCubbie with TokenPOSCubbie with TokenNerCubbie with TokenLemmaCubbie,
        () => new SentenceCubbie with SentenceParseCubbie,
        () => new TokenSpanCubbie),
      (d: DocCubbie) => Seq(Seq(d.name))
    ) with LazyCubbieConverter[DocCubbie]
    val mentions = new MongoCubbieCollection(collmention, () => new Mention, (m: Mention) => Seq(Seq(m.entity), Seq(m.docId)))


    val spdb = new SelfPredictingDatabase()
    spdb.numComponents = 5
    spdb.numArgComponents = 2


    //load query entities
    val devQueryEntities = loadSFEntitySeq(conf.getString("slot-data.query-dev")).map(e => e.tacId() -> e).toMap
    val trainQueryEntities = loadSFEntitySeq(conf.getString("slot-data.query-train")).map(e => e.tacId() -> e).toMap

    logger.info("Loaded Dev Entities:   " + devQueryEntities.size)
    logger.info("Loaded Train Entities: " + trainQueryEntities.size)

    val canonical2Entity = new HashMap[String, Entity]

    canonical2Entity ++= devQueryEntities.map(pair => pair._2.canonical() -> pair._2)
    canonical2Entity ++= trainQueryEntities.map(pair => pair._2.canonical() -> pair._2)


    //load relations from SF training annotation
    val trainRelations = loadSFAnnotation(conf.getString("slot-data.annotations-train"),
      id => trainQueryEntities(id),
      (docId, phrase) => canonical2Entity.getOrElseUpdate(phrase, new Entity().canonical(phrase).Id(new ObjectId))
    ).toSeq

    //    println(trainRelations.mkString("\n"))

    //load relations from SF dev annotation
    val devRelations = loadSFAnnotation(conf.getString("slot-data.annotations-dev"),
      id => devQueryEntities(id),
      (docId, phrase) => canonical2Entity.getOrElseUpdate(phrase, new Entity().canonical(phrase).Id(new ObjectId))
    ).toSeq

    val allRelations = trainRelations ++ devRelations
    val relationsByTuple = allRelations.groupBy(_.tuple)

    val matchedRelInstances = new HashSet[RelationInstance]

    val processedDocs = new HashSet[String]

    //load textual rows
    extractMentionPairFeaturesAndAddCandidateSlots(spdb, documents.iterator.limit(100), mentions,
      (doc, mention) => {
        processedDocs += doc.name.substring(doc.name.lastIndexOf("/") + 1, doc.name.lastIndexOf("."))
        canonical2Entity.getOrElseUpdate(mention.phrase(),
          new Entity().canonical(mention.phrase()).Id(new ObjectId()))
      },
      entity => {
        devQueryEntities.isDefinedAt(entity.tacId())
      },
      (src, dst, slot) => {
        val relations = relationsByTuple.get(Seq(src, dst)).map(_.filter(_.rel == slot))
        val gold = relations.map(_.size > 0).getOrElse(false)
        matchedRelInstances ++= relations.getOrElse(Seq.empty)
        gold
      }
    )

    logger.info("Matched Relation Instances: %d / %d".format(matchedRelInstances.size, allRelations.size))

    println("Missing:")
    for (rel <- allRelations; if (processedDocs(rel.provenance.docId) && !matchedRelInstances(rel))) {
      println(rel)
    }

    //find docs with relations that weren't matched
    val matchDebugOut = new PrintStream("match-debug.txt")
    for (doc <- documents.iterator.limit(100)) {
      val docId = doc.name().substring(doc.name().lastIndexOf("/") + 1, doc.name().lastIndexOf("."))
      val relations = allRelations.filter(_.provenance.docId == docId)
      val missing = relations.filter(r => r.state && !matchedRelInstances(r))
      if (missing.size > 0) {
        matchDebugOut.println(doc.string())
        for (r <- missing) {
          matchDebugOut.println("*****")
          matchDebugOut.println(r)
        }
        matchDebugOut.println("Mentions:")
        for (m <- mentions.query(_.docId(doc.name()))) {
          val entity = canonical2Entity(m.phrase())
          matchDebugOut.println(m.phrase())
          matchDebugOut.println(entity)
        }

      }
    }

    val id2entity = canonical2Entity.values.groupBy(_.id)

    //learn
    spdb.runLBFGS()

    spdb.evaluate(0.5).debug(System.out)

    spdb.debugTuples(spdb.tuples.take(1000), new PrintStream("tuples.txt"), id => id2entity(id).head.canonical())
    spdb.debugModel(new PrintStream("model.txt"))

    //load eval annotation (if available)
    //load training annotation

    //load documents
    //load structured sources


  }

  def loadSFEntitySeq(file: String): Seq[Entity] = {
    val article = XML.loadFile(file)
    val loaded = for (query <- article \\ "kbpslotfill" \\ "query") yield {
      val id = (query \ "@id").text
      val name = (query \ "name").text
      val entity = new Entity
      entity.tacId := id
      entity.canonical := name
      entity.id = id
      entity
    }
    loaded
  }

  case class Provenance(docId: String, charBegin: Int, charEnd: Int, phrase: String)

  case class RelationInstance(rel: String, tuple: Seq[Entity],
                              state: Boolean = true, provenance: Provenance)

  def loadSFAnnotation(annotationFile: String,
                       id2Entity: String => Entity,
                       docPhrase2Entity: (String, String) => Entity) = {

    var source = Source.fromFile(annotationFile)
    for (line <- source.getLines().drop(1);
         fields = line.split("\t");
         label = fields(3); if (targetSlots(label))) yield {
      val sfid = fields(1)
      val docId = fields(4)
      val phrase = fields(7)
      val arg2 = docPhrase2Entity(docId, phrase)
      val arg1 = id2Entity(sfid)
      val state = fields(10) match {
        case "1" => true
        case _ => false
      }
      val charBegin = fields(5).toInt
      val charEnd = fields(6).toInt
      RelationInstance(label, IndexedSeq(arg1, arg2), state, Provenance(docId, charBegin, charEnd, phrase))
    }
  }


}

object Conf {

  lazy val targetSlots = conf.getString("target-slots").split(",").map(_.trim).toSet

  val addedResources = new ArrayBuffer[String]

  lazy val parentOutDir = {
    val dir = new File(if (conf.hasPath("outDir")) conf.getString("outDir") else "out")
    dir.mkdirs()
    dir
  }



  lazy val outDir = {
    val date = Calendar.getInstance()
    val day = date.get(Calendar.DAY_OF_MONTH)
    val month = date.get(Calendar.MONTH)
    val year = date.get(Calendar.YEAR)
    val hour = date.get(Calendar.HOUR_OF_DAY)
    val min = date.get(Calendar.MINUTE)
    val sec = date.get(Calendar.SECOND)
    val ms = date.get(Calendar.MILLISECOND)

    val dayDir = new File(parentOutDir, "%d_%d_%d".format(day, month, year))
    val msDir = new File(dayDir, "run_%d_%d_%d_%d".format(hour, min, sec, ms))

    val latest = new File(parentOutDir, "latest")
    if (latest.exists()) {
      latest.delete()
    }
    msDir.mkdirs()

    for (resource <- addedResources) {
      val stream = getClass.getResourceAsStream("/" + resource)
      val in = Channels.newChannel(stream)
      val out = new FileOutputStream(new File(msDir,resource)).getChannel
      out.transferFrom(in,0l,Long.MaxValue)
      out.close()
      in.close()
    }

    Runtime.getRuntime.exec("/bin/ln -s %s %s".format(msDir.getAbsolutePath, latest.getAbsolutePath))

    msDir
  }

  private var _conf: Config = ConfigFactory.parseResources("default.conf")

  def add(resources: String) {
    addedResources += resources
    _conf = ConfigFactory.parseResources(resources).withFallback(conf)
    _conf.resolve()
  }

  def conf = _conf
}

