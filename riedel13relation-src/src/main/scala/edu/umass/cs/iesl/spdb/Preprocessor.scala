package edu.umass.cs.iesl.spdb

import cc.factorie.app.nlp._
import java.io._
import parse.ParseTree
import xml.XML
import cc.factorie.db.mongo.MongoCubbieCollection
import com.mongodb.{ DBCollection, Mongo}
import cc.factorie.app.strings.Stopwords
import collection.mutable.{ArrayBuffer, HashSet, HashMap}
import cc.refectorie.user.jzheng.coref.PosTag
import io.Source
import cc.factorie.db.mongo.MongoCubbieImplicits._

/**
 * @author sriedel
 */
object Preprocessor {

//  val mongoConn = new Mongo("localhost",27007)
//  val mongoDB = mongoConn.getDB("tackbpCubbieCoref")   // web data for tac-kbp
//  var colldoc = mongoDB.getCollection("documents")
//  var collmention = mongoDB.getCollection("mentions")
//  val collentity = mongoDB.getCollection("entities")
//  val collmentionPairs = mongoDB.getCollection("mentionpairs")

  def main(args: Array[String]) {
    //create factorie NLP objects here
    val dir = if(args.length > 0) args(0) else "/Users/lmyao/Documents/ie_ner/unsuprel/tackbp/LTW_ENG_20070309.0062.LDC2009T13.sgm" //"/iesl/canvas/dietz/tac/TAC_2010_KBP_Source_Data/data"
    val output = if(args.length > 1) args(1) else "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp/docs.triples.dat"
    val vfile = "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp/visualization.txt"
    val docidfile = "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp/docid_orig.txt"


      //for debugging
/* val doc = fromFile(new File(dir))
 CoreNLPTokenizer.annotate(doc)


for(sentence <- doc.sentences) {

   println("Sen:\n" + sentence.string)
   for(token <- sentence.tokens){
     println(token.string + "(" + token.stringStart + "-" + token.stringEnd + ")")
   }
} */


    val configFile =  "tackbp-lmyao-1.conf"
    Conf.add(configFile)

    //get all documents which contain any of the query entities
    //preprocess documents
    val processed = Util.loadHashSet(Conf.conf.getString("source-data.docidlist"))
    processDocs(Conf.conf.getString("source-data.dir"), processed)
    extractMentions()

//    val os = new PrintStream(output)
//    processed.map(x => os.println(x))
//    os.close
//     visualizeDocs(new File(Conf.outDir, "visualization.txt"))

  }
  def fromDirectory(dir:File): Seq[Document] = {
    for (file <- Util.files(dir)) yield fromFile(file)
  }

  //sentence segmentation and tokenization
  def fromFile(file:File) : Document =  {
//    val article = XML.loadFile(file)
//    val body = article \\ "DOC" \\ "BODY" \\ "TEXT"
//    val str = normalize(body.text)
//    val source = Source.fromFile(file)
//    var str = source.getLines().mkString("\n")
    val str = Util.readFile(file)
    val docname = file.getCanonicalPath.split("/").reverse(0).replace(".sgm", "")
    val document = new Document(docname, str)
    document
  }

/* //the data is too noisy, do some cleaning
  def normalize(source : String) : String = {
    //2008-09-25T21:33:06
    var str =  source.replaceAll("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}","")
    var res = ""
    val fields = str.split("\n")
    for(field <- fields){
      if(field.length == 0) res += "\n"
      else if(field.startsWith("http")) res += " "
      res += " " + field
    }
    res += "\n"
    res
  }  */
  
 /* def processDocs(dir : String,  processed : HashSet[String]){
    //if(processed.size == 0) coll.drop()
    val documents = Database.CorefDB.documents
    documents.drop
    //val docs = fromDirectory(new File(dir))
    val docs = for(file  <- Util.files(new File(dir)); if(processed.contains(file.getCanonicalPath.split("/").reverse(0).replace(".sgm", "")))) yield fromFile(file)

    val lemmaTagger = new CoreNLPMorphTagger
    var count = 0
    for(doc <- docs){
      CoreNLPTokenizer.annotate(doc)
      POS.annotate(doc)
      NERTagger.annotate(doc)
      lemmaTagger.annotate(doc)
      TigerParser.process(doc)

      val cd = new DocumentCubbie(() => new TokenCubbie with TokenPOSCubbie with TokenNerCubbie with TokenLemmaCubbie, () => new SentenceCubbie with SentenceParseCubbie, () => new TokenSpanCubbie)
      cd.storeDocument(doc)
      documents += cd
      processed.add(doc.name)
      count += 1
      if(count % 1000 == 0){
        println("Processed documents: " + count)
      }
      if(count == 10000) return
    }
  } */

  //only process docs which contain one of the query entities
  def processDocs(dir : String, processed : HashSet[String]){
    val documents = Database.CorefDB.documents
    
    val docids = documents.query(select = _.name.select).toSeq.map(_.name())
    println("Existing docs:" + docids.size)
    println("First doc:" + docids.take(1).mkString )
//    val docids = new HashSet[String]
//    for(doc <-  documents.query(select = _.name.select).toSeq.map(_.name)) docids += doc.name()
//    documents.drop
//
//    Database.CorefDB.entities.drop()
//    EntityInitializer.loadEntities()
//
//    val entities = Database.CorefDB.entities.filter(_.tacId.isDefined)
//    //var canonical2entities = entities.groupBy(_.canonical())
//    val canonicals = entities.map(_.canonical()) //all initial query entity names,
//    println("Query entities:" + canonicals.size)
    val docs = for(file  <- Util.files(new File(dir)) ; if(processed.contains(file.getCanonicalPath.split("/").reverse(0).replace(".sgm", "")) && !docids.contains(file.getCanonicalPath.split("/").reverse(0).replace(".sgm", "")))) yield fromFile(file)

    val lemmaTagger = new CoreNLPMorphTagger
    var count = 0
    for(doc <- docs; if(doc.string.length()>0) ){                //; if( processed.contains(doc.name) || Util.checkSource(doc.string,canonicals) )
      println("Process doc " + doc.name)
      CoreNLPTokenizer.annotate(doc)
      POS.annotate(doc)
      NERTagger.annotate(doc)
      lemmaTagger.annotate(doc)
      TigerParser.process(doc)

      val cd = new DocumentCubbie(() => new TokenCubbie with TokenPOSCubbie with TokenNerCubbie with TokenLemmaCubbie, () => new SentenceCubbie with SentenceParseCubbie, () => new TokenSpanCubbie)
      cd.storeDocument(doc)
      documents += cd
      count += 1
      if(count % 1000 == 0){
        println("Processed documents: " + count)
      }
    }
  }
  
  def extractMentions() {
    val mentions = Database.CorefDB.mentions
    mentions.drop()

    val documents = Database.CorefDB.documents
    for(cd <- documents){
      val doc = cd.fetchDocument
      for(sentence <- doc.sentences){
        EntityMentionUtils.findAllSimpleMentions(sentence,mentions)
      }
      if(mentions.size % 100 == 0) println("Extracted mentions: " + mentions.size )
    }
    println("Extracted total mentions: " + mentions.coll.getCount)
  }

  def extractMentionPairs(){
    val documents = Database.CorefDB.documents

    val mentions = Database.CorefDB.mentions
    val mentionpairs = new MongoCubbieCollection(Database.CorefDB.coll("mentionPairs"), () => new MentionTuple)

    for(cd <- documents){
      val doc = cd.fetchDocument
      val mentionsOfDoc = mentions.query(_.docId(doc.name))

      val mentionSeq = new ArrayBuffer[Mention]
      for(mention <- mentionsOfDoc) {
        mentionSeq += mention
      }

      for(source <- mentionSeq; dest <- mentionSeq;  if(doc.tokens(source.tokenBegin()).sentence.start == doc.tokens(dest.tokenBegin()).sentence.start   //the same sentence
        && source.phrase() != dest.phrase() && source.tokenEnd() <= dest.tokenBegin() && doc.tokens(source.tokenBegin()).sentence.size < 100)){
        val pair = new MentionTuple
        pair.mentions := Seq(source,dest)
        mentionpairs += pair

        if(mentionpairs.coll.getCount % 100 == 0) println("Extracted mention pairs: " + mentionpairs.coll.getCount )
      }
    }
    println("Extracted total mention pairs: " + mentions.coll.getCount)
  }

  def visualizeDocs(output : File){

    val documents = Database.CorefDB.documents
    val mentions = Database.CorefDB.linkedMentions
    var tokenFreqMap = new HashMap[String,Int]

    println("Number of docs:" + documents.size)
    val os = new PrintStream(output)
    for(cd <- documents; if(cd.name().compare("NYT_ENG_20080805.0089.LDC2009T13") == 0)){           //; if(cd.name().compare("eng-NG-31-104102-11676816") == 0)
      val doc = cd.fetchDocument
      os.println("Doc " + doc.name + " with sentences: " + doc.sentences.size)
      val mentionsOfDoc = mentions.query(_.docId(doc.name)).toSeq
      val docRelMentions = Database.KBDB.relMentions.query(_.docId(doc.name)).toSeq
      val arg12relMentions = docRelMentions.groupBy(_.arg1Mention())
      val id2Mentions = mentionsOfDoc.groupBy(_.id)

      for(sentence <- doc.sentences) {
        os.println("Sen:" + sentence.string)
        for(token <- sentence.tokens){
          os.println(token.string + "(" + token.stringStart + "-" + token.stringEnd + ")")
        }
        os.println(sentence.attr[ParseTree])
        val sent2Mentions = mentionsOfDoc.groupBy(m => FactorieTools.sentence(doc, m))
        val sentMentions = sent2Mentions.getOrElse(sentence, Seq.empty)
        for(mention : Mention <- sentMentions){
          os.println("Mention: " + mention.docId() + "\t"  +  mention.phrase() + "\t" + mention.canonical()  )
        }
      /*  os.println("Relation Mentions:")
        for (arg1 <- sentMentions) {
          os.println("Arg1 Mention: " + arg1.phrase())
          for ((arg2id, rms) <- arg12relMentions.getOrElse(arg1.id, Seq.empty).groupBy(_.arg2Mention())) {
            val arg2 = id2Mentions(arg2id).head
            os.println("Arg2 Mention: " + arg2.phrase())
//            for (rm <- rms) {
//              os.println("Feat:         " + rm.label())
//            }
          }
        }*/
        
        os.println("=====================")
      }
      




      //calculate token frequency
    /*  val localStat = doc.tokens.groupBy(_.string.trim.toLowerCase.replaceAll("\\n", " ")).mapValues(_.size)
      localStat.foreach(tf => {
        if(tokenFreqMap.contains(tf._1)) {
          val ocount = tokenFreqMap(tf._1)
          tokenFreqMap.update(tf._1,ocount+tf._2)
        }else tokenFreqMap += tf._1 -> tf._2
      }) */

    } 
    
    
    
    os.close

  /*  val out = new java.io.PrintWriter(output + ".tokenFreq")
    tokenFreqMap.foreach(tf => out.write(tf._1 + "|" + tf._2 + "\n"))
    out.flush
    out.close  */

  }

}

