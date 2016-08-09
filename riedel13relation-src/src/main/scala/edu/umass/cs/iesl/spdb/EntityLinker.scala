package edu.umass.cs.iesl.spdb

import Conf._
import xml.XML
import org.bson.types.ObjectId
import cc.factorie.db.mongo.MongoCubbieImplicits._
import io.Source
import collection.mutable.HashMap


/**
 * @author sriedel
 */
object EntityLinker extends HasLogger {
  def main(args: Array[String]) {
    Conf.add(args(0))
    linkMentions()
  }


  def linkMentions() {
    Database.CorefDB.linkedMentions.drop()
    //iterate over all mentions, and link them to entities
    //naive method: load all entities into memory
    val entities = Database.CorefDB.entities.toSeq
    var canonical2entities = entities.groupBy(_.canonical())

    for (doc <- Database.CorefDB.documents.query(select = _.name.select)) {
      logger.info("Processing " + doc.name())
      for (mention <- Database.CorefDB.mentions.query(_.docId(doc.name()))) {
        val phrase = mention.phrase()
        canonical2entities.get(phrase) match {
          case Some(candidates) =>
            mention.entity ::= candidates.head
          case None =>
            val id = new ObjectId
            val newEntity = new Entity().canonical(phrase).Id(id)
            mention.entity ::= newEntity
            Database.CorefDB.entities += newEntity
            canonical2entities = canonical2entities + (phrase -> Seq(newEntity))
        }
        Database.CorefDB.linkedMentions += mention
      }
    }
  }
}

object EntityInitializer extends HasLogger {


  def loadSFEntitySeq(file: String): Seq[(Entity, Mention)] = {
    val article = XML.loadFile(file)
    val loaded = for (query <- article \\ "kbpslotfill" \\ "query") yield {
      val id = (query \ "@id").text
      val name = (query \ "name").text
      val tacType = (query \ "enttype").text
      val docid = (query \ "docid").text


      val entity = new Entity
      entity.tacId := id
      entity.canonical := name
      entity.tacType := tacType
      entity.id = id
      val mention = new Mention
      mention.docId := docid
      entity -> mention
    }
    loaded
  }

  def loadSFEntitySeqWithFreebase(file: String, freebaseFile : String): Seq[(Entity, Mention)] = {

    val source = Source.fromFile(freebaseFile)
    val canonical2line = new HashMap[String, String]
    for (line <- source.getLines()){
      val fields = line.split("\t")
      canonical2line += fields(1) -> line
    }

    val article = XML.loadFile(file)
    val loaded = for (query <- article \\ "kbpslotfill" \\ "query") yield {
      val id = (query \ "@id").text
      val name = (query \ "name").text
      val tacType = (query \ "enttype").text
      val docid = (query \ "docid").text


      val entity = new Entity
      entity.tacId := id
      entity.canonical := name
      entity.tacType := tacType
      entity.id = id
      if(canonical2line.contains(name)){
        entity.freebaseId :=  canonical2line(name).split ("\t")(0)
        entity.wikipediaId :=  canonical2line(name).split ("\t")(2)
      }
      val mention = new Mention
      mention.docId := docid
      entity -> mention
    }
    loaded
  }
  
  def loadFreebaseSlotEntities(file : String) = {
    val mid2entity = Database.CorefDB.entities.filter(_.freebaseId.isDefined).groupBy(_.freebaseId())
    val source = Source.fromFile(file)
    for (line <- source.getLines()){
      val fields = line.split("\t")
      if(!mid2entity.contains(fields(0))) {
        val id = new ObjectId
        val newEntity = new Entity().canonical(fields(1)).Id(id).freebaseId(fields(0)).wikipediaId(fields(2))
        Database.CorefDB.entities += newEntity
      }
    }
  }

  def main(args: Array[String]) {
    Conf.add(args(0))

    //load entities from KBP data
    Database.CorefDB.entities.drop()
    loadEntities()

//    for(ent : Entity <- Database.CorefDB.entities){
//      print(ent.canonical() + "\t" )
//      if(ent.freebaseId.isDefined) print(ent.freebaseId()+"\t")
//      if(ent.wikipediaId.isDefined) print(ent.wikipediaId()+"\t")
//      if(ent.tacId.isDefined) print(ent.tacId())
//      println
//    }
  }

  def loadEntities() {
    logger.info("Loading KBP query files")
    val devQueryEntities = loadSFEntitySeqWithFreebase(conf.getString("slot-data.query-dev"), conf.getString("freebase.query")) //loadSFEntitySeq(conf.getString("slot-data.query-dev"))
    val trainQueryEntities = loadSFEntitySeqWithFreebase(conf.getString("slot-data.query-train"), conf.getString("freebase.query")) //loadSFEntitySeq(conf.getString("slot-data.query-train"))

    trainQueryEntities.foreach(_._1.tacTest := false)
    devQueryEntities.foreach(_._1.tacTest := true)

    //add entities to entity collection.
    logger.info("Adding KBP entities to database")
    //todo: this should align with existing entities
    Database.CorefDB.entities ++= devQueryEntities.map(_._1)
    Database.CorefDB.entities ++= trainQueryEntities.map(_._1)

    loadFreebaseSlotEntities(conf.getString("freebase.slots"))
  }

}

object KBPSlotLoader extends HasLogger {

  def main(args: Array[String]) {
    Conf.add(args(0))
    //load slots from KBP data
    loadSlots()

  }

  def loadSlots() {
    logger.info("Loading KBP slot annotation")
    loadSFAnnotation(conf.getString("slot-data.annotations-train"))
    loadSFAnnotation(conf.getString("slot-data.annotations-dev"))
  }


  def loadSFAnnotation(annotationFile: String) {

    var arg2Matched = 0

    var source = Source.fromFile(annotationFile)
    val lines = (for (line <- source.getLines().drop(1);
                     fields = line.split("\t");
                     label = fields(3); if (targetSlots(label))) yield fields).toSeq
    val docs = lines.map(_.apply(4)).toSet.toSeq
    val doc2mentions = Database.CorefDB.linkedMentions.query(_.docId.valuesIn(docs)).toSeq.groupBy(_.docId())
    val sfids = lines.map(_.apply(1)).toSet.toSeq
    val sfid2entities = Database.CorefDB.entities.query(_.tacId.valuesIn(sfids)).toSeq.groupBy(_.tacId())

    for (fields <- lines) {
      val label = fields(3)
      val sfid = fields(1)
      val docId = fields(4)
      val charBegin = fields(5).toInt
      val charEnd = fields(6).toInt
      val phrase = fields(7)
      val state = fields(10) match {
        case "1" => true
        case _ => false
      }
      val arg1 = sfid2entities(sfid).head
      //find first mention that overlaps
      val mention = new Mention().phrase(phrase).docId(docId).charBegin(charBegin).charEnd(charEnd)
      val mentions = doc2mentions.getOrElse(docId,Seq.empty)
      for (overlap <- mentions.find(_.overlap(mention))) {
        val arg2Id = overlap.entity()
        mention.entity := arg2Id
        if (state) {
          val rel = new KBRelation().arg1(arg1.id).arg2(arg2Id).label(label).tacTest(arg1.tacTest())
          rel.tacProvenance := mention
          Database.KBDB.relations += rel
          Database.KBDB.tacRelations += rel
          Database.KBDB.tacMentions += mention
        }
        arg2Matched += 1
      }
    }
    logger.info("Loaded slots:  " + lines.size)
    logger.info("Matched slots: " + arg2Matched)

  }

}

object FreebaseLoader extends HasLogger{

  def main(args: Array[String]) {
    Conf.add(args(0))
    Database.KBDB.relations.drop()
    //load relations from Freebase
    loadFreebase()
//    val id2entity = Database.CorefDB.entities.groupBy(_.id)
//    for(relation <- Database.KBDB.relations){
//      val arg1 : Entity = id2entity(relation.arg1()).head
//      val arg2 : Entity = id2entity(relation.arg2()).head
//      println(arg1.canonical() + "\t" + relation.label() + "\t" + arg2.canonical())
//    }
  }

  def loadFreebase(){
    loadFreebaseRelations(Conf.conf.getString("freebase.relation"))
  }

  def loadFreebaseRelations(relationFile : String)  {
    val mid2entity = Database.CorefDB.entities.filter(_.freebaseId.isDefined).groupBy(_.freebaseId())
    val source = Source.fromFile(relationFile)
    for(line <- source.getLines())  {
      val fields = line.split("\t")
      val midsrc = fields(0)
      val label = fields(1)
      val middst = fields(2)
      val arg1 = if (mid2entity.contains(midsrc)) mid2entity(midsrc).head else null
      val arg2 = if (mid2entity.contains(middst)) mid2entity(middst).head else null
      if(arg1 != null && arg2 != null){
        val rel = new KBRelation().arg1(arg1.id).arg2(arg2.id).label(label).freebase(true)
        Database.KBDB.relations += rel
      }
    }
  }
}

object Experiment extends HasLogger {
  def main(args: Array[String]) {
    Conf.add(args(0))
    Conf.add(args(1))
    logger.info("Starting experiment pipeline for slots: " + Conf.targetSlots.mkString(","))
    if (conf.getBoolean("experiment.clearKBDB")) {
      logger.info("Clearing Database...")
      Database.KBDB.dropDatabase()
    }
    if (conf.getBoolean("experiment.initEntities")) EntityInitializer.loadEntities()
    if (conf.getBoolean("experiment.linkMentions")) EntityLinker.linkMentions()
    if (conf.getBoolean("experiment.knowledgebase")) KnowledgeBase.entityLinking()     //this takes very long time to run, usually run it separately from others, Limin Yao
    if (conf.getBoolean("experiment.loadSlots")) KBPSlotLoader.loadSlots()
    if (conf.getBoolean("experiment.loadFreebase")) FreebaseLoader.loadFreebase
    if (conf.getBoolean("experiment.evalMentions")) MentionEval.evalMentions()
    if (conf.getBoolean("experiment.extractRelationMentions")) RelationMentionExtractor.extractRelationMentions1()
    if (conf.getBoolean("experiment.runPCA")) PCASlotFiller.runPCA()
  }
}
