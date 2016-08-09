package edu.umass.cs.iesl.spdb
import cc.factorie.app.nlp._
import java.io._
import scala.io.Source
import xml.{XML, NodeSeq}
import com.mongodb.{ DBCollection, Mongo}
import cc.factorie.app.strings.Stopwords
import collection.mutable.{HashMap, ArrayBuffer, HashSet}
import org.bson.types.ObjectId
import cc.refectorie.user.jzheng.coref.util.Options
import cc.refectorie.user.jzheng.coref.{CorefLabel, Scorer}
import cc.refectorie.user.jzheng.coref.PosTag
import cc.factorie.app.nlp.coref.PairwiseMention
import cc.factorie.db.mongo.MongoCubbieImplicits._
import Conf._

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/27/12
 * Time: 11:04 AM
 * To change this template use File | Settings | File Templates.
 */

object KnowledgeBase {

  val PREFIX = "^(mrs|mrs\\.|miss|ms|ms\\.|madam|maam|mr|mr\\.|sir|dr|dr\\.) .*"
  val SUFFIX = ".* (co|co\\.|inc|inc\\.)$"

  def main(args: Array[String]) {
    val configFile =  "tackbp-lmyao-1.conf"
    Conf.add(configFile)

    //entityLinking()

    tupleLinking()
  }

  def entityLinking(){
//    Database.CorefDB.entities.drop()
//    EntityInitializer.loadEntities()    //now this loads both tac and Freebase entities
//    Database.CorefDB.linkedMentions.drop()
    linkMentions
  }

  def tupleLinking(){
    val slotMap = loadSFAnnotation(conf.getString("slot-data.query-train"),conf.getString("slot-data.annotations-train"),conf.getString("slot-data.query-dev"),conf.getString("slot-data.annotations-dev"))
    linkTAC(slotMap,new File(Conf.outDir, "tac_link.txt"))
  }
  
 /* //create sfentities collection, storing all slot filling queries, /iesl/canvas/dietz/tac/2011_train_reg_sf/TAC_2011_KBP_English_Training_Regular_Slot_Filling_Annotation/data/tac_2010_kbp_training_slot_filling_queries.xml
  def loadSFEntities(coll : DBCollection, file: String) {
    coll.drop()
    val entities = new MongoCubbieCollection(coll, () => new Entity, (e:Entity) => Seq(Seq(e.tacId)))
    val loaded: Seq[Entity] = loadSFEntitySeq(file)
    entities ++= loaded
  }    */


  def loadSFEntitySeq(file: String): Seq[Entity] = {
    val article = XML.loadFile(file)
    val loaded = for (query <- article \\ "kbpslotfill" \\ "query") yield {
      val id = (query \ "@id").text
      val name = (query \ "name").text
      val entity = new Entity
      entity.tacId := id
      entity.canonical := name
      entity
    }
    loaded
  }

  def loadSFRelations(coll : DBCollection, file : String){}

  def load(file: String, entMap : HashMap[String, String]) {
 
    val article = XML.loadFile(file)
    for(query <- article \\ "kbpslotfill" \\ "query"){
      val id = (query \ "@id").text
      val name = (query \ "name").text
      entMap += id -> name
    }

  }
  
  def loadSFAnnotation(trainQueryfile : String, trainTuplefile : String, testQueryfile : String,  testTuplefile : String) : HashMap[String, HashMap[String,String]] = {
    val trainentitiesMap = new HashMap[String, String]
    val testentitiesMap = new HashMap[String, String]
    load(trainQueryfile,trainentitiesMap)
    load(testQueryfile,testentitiesMap)

    val slotsMap = new HashMap[String,HashMap[String,String]]
    var source = Source.fromFile(trainTuplefile)
    for(line <- source.getLines(); if(!line.contains("sf_id") && !line.endsWith("-1"))){
      val fields = line.split("\t")
      val sfid = fields(1)
      val label = fields(3)
      val arg2 = fields(7)
      val arg1 = trainentitiesMap(sfid)
      val slots = slotsMap.getOrElseUpdate(arg1,new HashMap[String,String])
      if(arg2.length()>1)
        slots += (arg2->label)
      slots += ("train"->"train")
    }
    source = Source.fromFile(testTuplefile)
    for(line <- source.getLines(); if(!line.contains("sf_id") && !line.endsWith("-1") ) ){
      val fields = line.split("\t")
      val sfid = fields(1)
      val label = fields(3)
      val arg2 = fields(7)
      val arg1 = testentitiesMap(sfid)
      val slots = slotsMap.getOrElseUpdate(arg1,new HashMap[String,String])
      if(arg2.length()>1)
        slots += (arg2->label)
      slots += ("test"->"test")
    }
    slotsMap
  }

  //coreference, link query entities and slots, todo: fix slot linking
  def linkTAC( slotsMap : HashMap[String, HashMap[String,String]], output:File){

    val nerMap = new HashMap[String, String]()
    nerMap += "PER" -> "PERSON"
    nerMap += "ORG" -> "ORGANIZATION"

    val matched = new HashSet[String]
    val uos = new PrintStream(output + ".unmatched")

    val mentions = Database.CorefDB.linkedMentions

    val os = new PrintStream(output)
    val entityStrs = slotsMap.keys

    for(cd <- Database.CorefDB.documents ){
      val doc = cd.fetchDocument
      os.println("#Document:" + doc.name)
      
      val mentionSeq = new ArrayBuffer[Mention] //including the original mentions and additional matched mentions
      val matchedMentions = new HashSet[(Int,Int)]

    /*  for (entstr <- entityStrs){     //this is incorrect for number slots, skip number slots
        //find whether any entity occur
        val abbr = entstr.split("[-\\s]").map(_.substring(0,1)).mkString("")
        if(doc.string.contains(entstr) || (abbr.matches("^[A-Z]+$") && doc.string.contains(abbr))){
          //match tac annotations
          val slots = slotsMap(entstr).keys
          for (sentence <- doc.sentences){

            for(slot <- slots; if (slot != "train" && slot != "test")){
              if(sentence.string.contains(slot)){
                var dest  : Mention = null

                var charBegin =  doc.string.indexOf(slot, sentence.tokens.head.stringStart)
                var charEnd = charBegin + slot.length

                var tokenEnd = sentence.tokenAtCharIndex(charEnd-1).position
                var tokenBegin = if (!slot.contains(" ")) tokenEnd else sentence.tokenAtCharIndex(charBegin).position

                if(charBegin != -1 && charEnd != -1)  {
                  dest = new Mention
                  dest.charBegin := charBegin
                  dest.charEnd := charEnd
                  dest.tokenBegin := tokenBegin
                  dest.tokenEnd := tokenEnd
                  dest.phrase := doc.string.substring(dest.charBegin(),dest.charEnd())
                  dest.canonical := slot

                  if(!matchedMentions.contains((dest.charBegin(),dest.charEnd()))) {
                    mentionSeq += dest
                    matchedMentions += charBegin -> charEnd
                  }
                }

              }
            }
          }
        }
      }*/

      //get original mentions
      val mentionsOfDoc = mentions.query(_.docId(doc.name)).toSeq
      for(mention <- mentionsOfDoc){
        if(!matchedMentions.contains((mention.charBegin(),mention.charEnd()))){
          mentionSeq += mention
          matchedMentions += mention.charBegin() -> mention.charEnd()
        }
      }

      val entityIds = mentionsOfDoc.map(_.entity())
      val id2entities = Database.CorefDB.entities.query(_.idsIn(entityIds)).toArray.groupBy(_.id)
      
      var prefix = "other"
      val sent2Mentions = mentionsOfDoc.groupBy(m => FactorieTools.sentence(doc, m))
      for(sentence <- doc.sentences; if(sentence.size < 100)){
        val candidateMentions = sent2Mentions.getOrElse(sentence, Seq.empty)
        for(arg1 : Mention <- candidateMentions; arg2 : Mention <- candidateMentions; if( arg1.canonical()!="None" && arg2.canonical() != "None"
          && arg1.canonical() != arg2.canonical()  && arg1.tokenEnd() <= arg2.tokenBegin())){

          val arg1Ent : Entity = id2entities(arg1.entity()).head
          val arg2Ent : Entity = id2entities(arg2.entity()).head
          
          var srcner = if(arg1Ent.tacType.isDefined) nerMap(arg1Ent.tacType()) else if(arg1.label.isDefined) arg1.label() else "NONE"
          var dstner = if(arg2Ent.tacType.isDefined) nerMap(arg2Ent.tacType()) else if(arg2.label.isDefined) arg2.label() else "NONE"


          val features = Features.extractFeatures(arg1,arg2,doc, srcner, dstner)

          if(slotsMap.contains(arg1Ent.canonical())){
            if(slotsMap(arg1Ent.canonical()).contains("train") )   prefix = "train"
            else prefix = "test"
          } else if(slotsMap.contains(arg2Ent.canonical())){
            if (slotsMap(arg2Ent.canonical()).contains("train"))  prefix = "train"
            else prefix = "test"
          }else 
            prefix = "other"

          if(prefix.compare("other") != 0){
            if (slotsMap.contains(arg1Ent.canonical())  ){
              if (slotsMap(arg1Ent.canonical()).contains(arg2Ent.canonical())) {
                prefix += ":" + slotsMap(arg1Ent.canonical())(arg2Ent.canonical())
                matched += arg1Ent.canonical()+"|"+arg2Ent.canonical()
              }
            }else if(slotsMap(arg2Ent.canonical()).contains(arg1Ent.canonical())){
              prefix += ":" + slotsMap(arg2Ent.canonical())(arg1Ent.canonical())
              matched += arg2Ent.canonical()+"|"+arg1Ent.canonical()
            }
          }
          
//          if(prefix.contains(":") && features.size == 0){
//            uos.println("Missing data:")
//            uos.println("Sen:" + sentence.string)
//            uos.println(arg1.phrase() + "\t" + arg2.phrase())
//          }

          if(features.size > 0){
            os.println(prefix + "\t" + arg1Ent.canonical() + "|" + arg2Ent.canonical() + "\t"  + features.mkString("\t") + "\tsen#" + doc(arg2.tokenBegin()).sentence.tokens.map(_.string).mkString(" "))
          }
        }
      }

      
    /*  for(arg1 <- mentionSeq; arg2 <- mentionSeq;  if(doc(arg1.tokenBegin()).sentence.start == doc(arg2.tokenBegin()).sentence.start
        && arg1.phrase()!="None" && arg2.phrase() != "None"  && arg1.phrase() != arg2.phrase()  && arg1.tokenEnd() <= arg2.tokenBegin()
        
        && doc.tokens(arg1.tokenBegin()).sentence.size < 100)){

      } */
  

    }

        for(entstr <- slotsMap.keys){
          val slots = slotsMap(entstr).keys
          for(slot <- slots; if(slot != "train" && slot != "test")){
            if(!matched.contains(entstr+"|"+slot))
             uos.println("Missing:" + entstr + "\t" + slot)
          }
        }

    os.close
    uos.close

  }

  /*
  coreference and link to query entities , canonical is important, it is the keyword for creating entities for the mentions
  mentions: all ner mentions, phrase is original char sequence, canonical is normalization char sequence, label is ner
  additional mentions:
     query entities or their string variations:  phrase is original char sequence, canonical is query entity format, label now is undefined,
     common noun, pronoun, label undefined, canonical="None", phrase is original char seq

  for coreference, see Coref, for each mention in a cluster, its canonical is modified to be the cluster's canonical string

  heuristic linking to tac entities:
    -line break   Peter D. Hart \n Research Associates
    -capital, lower case
    -Wal-mart Walmart Wal-Mart
    -Dr. George Tiller vs. George Tiller
    -Peter D. Hart vs. Peter Hart
    -Massey Energy Co. vs. Massey Energy

  */
  def linkMentions{
    val docid2canonicals = Util.loadMultiMap(Conf.conf.getString("source-data.ranklist"))
    val entities = Database.CorefDB.entities.toSeq
    var canonical2entities = entities.groupBy(_.canonical())
    val canonicals = Util.loadHash(conf.getString("slot-data.tackbp-query")).values.toSeq
    //val canonicals = entities.map(_.canonical())
    //for coref
    val modelOnDisk = conf.getString("coref.modelOnDisk")//"/iesl/canvas/lmyao/workspace/spdb/duproth_perceptron300_falsefeaturevalues.model"    //on blake,jzheng's dir
    val clsf = Coref.loadClassifier(modelOnDisk, "Perceptron")
    val featureStyle = "DupRoth"
    val scorer = new Scorer[PairwiseMention](Options.predSemEval2010, Options.trueSemEval2010)
    CorefLabel.loadTokenFrequency(conf.getString("coref.tokenFreq"))

    var count = 0                    //Database.CorefDB.linkedMentions.query(_.docId(doc.name())).size == 0  &&APW_ENG_20080214.1020.LDC2009T13
    for (doc <- Database.CorefDB.documents ;if(Database.CorefDB.linkedMentions.query(_.docId(doc.name())).size == 0 && doc.name().compareTo("NYT_ENG_20070820.0056.LDC2009T13")!=0)) {      //;if(doc.name().compareTo("eng-NG-31-140748-12218616")==0)
      val candidates = docid2canonicals(doc.name())  //this contains all tac-queries which occur in this doc
      //logger.info("Processing " + doc.name())
      println("Processing " + doc.name())
      val mentionSeq = new ArrayBuffer[Mention] //including the original mentions and additional matched mentions
      val mentionIndexMap = new HashMap[String,Mention]

      val document = doc.fetchDocument

      for (mention <- Database.CorefDB.mentions.query(_.docId(doc.name())).toSeq; if(mention.label() != "NUMBER" && mention.label() != "DATE")) {
        if(!mentionIndexMap.contains((mention.charBegin()+"-"+mention.charEnd()))){
          mentionSeq += mention
          mentionIndexMap += (mention.charBegin()+"-"+mention.charEnd() ) -> mention
        }
      }

      //heuristic matching against tac queries
      //for (entstr : String <- canonicals){
      for (entstr : String <- candidates){
        //find whether any entity occur
        val abbr = Util.abbr(entstr)
    //    if(doc.string().contains(entstr) || doc.string().split("\n").mkString(" ").contains(entstr)   || (abbr.matches("^[A-Z]+$") && doc.string().contains(abbr))){
        
          for(sentence <- document.sentences){
            var offset2token : HashMap[Int, Token] = null
            var senstr = sentence.string.replaceAll("\n", " ")
            if(sentence.string.contains(entstr)  ){
              offset2token = new HashMap[Int,Token]
              for(token <- sentence.tokens){
                offset2token += token.stringStart -> token
                offset2token += token.stringEnd -> token
              }
              
              val charBegin = doc.string().indexOf(entstr,sentence.tokens.head.stringStart)
              val charEnd = charBegin + entstr.length
              val endToken = offset2token.getOrElse(charEnd,null)  //sentence.tokenAtCharIndex(charEnd).position
              val beginToken = offset2token.getOrElse(charBegin,null) //sentence.tokenAtCharIndex(charBegin).position
              if(charBegin > -1 && charEnd > -1 && beginToken != null && endToken != null && endToken.position >= beginToken.position){
                val source = new Mention
                source.docId :=document.name
                source.charBegin := charBegin
                source.charEnd := charEnd
                source.tokenBegin := beginToken.position
                source.tokenEnd := endToken.position
                source.phrase := document.string.substring(source.charBegin(),source.charEnd())
                source.canonical := entstr

                if(!mentionIndexMap.contains(source.charBegin()+"-"+source.charEnd())) {
                  mentionSeq += source
                  mentionIndexMap += (source.charBegin()+"-"+source.charEnd() ) -> source
                }else{
                  val origMention = mentionIndexMap(source.charBegin()+"-"+source.charEnd())
                  origMention.canonical := entstr
                }
              }
            } else if(senstr.contains(entstr)) {   //fix line break in one sentence

              offset2token = new HashMap[Int,Token]
              for(token <- sentence.tokens){
                offset2token += token.stringStart -> token
                offset2token += token.stringEnd -> token
              }

              val bias = sentence.tokens.head.stringStart
              val charBegin = senstr.indexOf(entstr) + bias
              val charEnd = charBegin + entstr.length
              val endToken = offset2token.getOrElse(charEnd,null)  //sentence.tokenAtCharIndex(charEnd).position
              val beginToken = offset2token.getOrElse(charBegin,null) //sentence.tokenAtCharIndex(charBegin).position
              if(charBegin > -1 && charEnd > -1 && beginToken != null && endToken != null && endToken.position >= beginToken.position){
                val source = new Mention
                source.docId :=document.name
                source.charBegin := charBegin
                source.charEnd := charEnd
                source.tokenBegin := beginToken.position
                source.tokenEnd := endToken.position
                source.phrase := document.string.substring(source.charBegin(),source.charEnd())
                source.canonical := entstr

                if(!mentionIndexMap.contains(source.charBegin()+"-"+source.charEnd())) {
                  mentionSeq += source
                  mentionIndexMap += (source.charBegin()+"-"+source.charEnd() ) -> source
                }
                else{
                  val origMention = mentionIndexMap(source.charBegin()+"-"+source.charEnd())
                  origMention.canonical := entstr
                }
              }
            }  else {   //fix Mai-Mai militia,   HOPE vs. Hope      bug      //entstr.split("[-\\s]").size > 1
              senstr = sentence.string.toLowerCase
              val bias = sentence.tokens.head.stringStart
              var entstr1 = entstr.replaceAll("-", " ").toLowerCase
              if(senstr.contains(entstr.toLowerCase) || senstr.contains(entstr1) ){

                offset2token = new HashMap[Int,Token]
                for(token <- sentence.tokens){
                  offset2token += token.stringStart -> token
                  offset2token += token.stringEnd -> token
                }

                var charBegin = if(senstr.contains(entstr.toLowerCase)) senstr.indexOf(entstr.toLowerCase)  else senstr.indexOf(entstr1)

                charBegin += bias
                val charEnd = charBegin + entstr.length
                val endToken = offset2token.getOrElse(charEnd,null)  //sentence.tokenAtCharIndex(charEnd).position
                val beginToken = offset2token.getOrElse(charBegin,null) //sentence.tokenAtCharIndex(charBegin).position

                if(charBegin > -1 && charEnd > -1 && beginToken != null && endToken != null && endToken.position >= beginToken.position){
                  val source = new Mention
                  source.docId :=document.name
                  source.charBegin := charBegin
                  source.charEnd := charEnd
                  source.tokenBegin := beginToken.position
                  source.tokenEnd := endToken.position
                  source.phrase := document.string.substring(source.charBegin(),source.charEnd())
                  source.canonical := entstr

                  if(!mentionIndexMap.contains(source.charBegin()+"-"+source.charEnd())) {
                    mentionSeq += source
                    mentionIndexMap += (source.charBegin()+"-"+source.charEnd() ) -> source
                  } else{
                    val origMention = mentionIndexMap(source.charBegin()+"-"+source.charEnd())
                    origMention.canonical := entstr
                  }
                }
              }else{
                for(token <- sentence.tokens){
                  if((token.string.compareTo(abbr) == 0 && abbr.matches("^[A-Z]+$")) || token.string.replaceAll ("-", "").toLowerCase.compare(entstr.toLowerCase)==0 ){    //abbr and Wal-mart
                    val source = new Mention
                    source.docId :=document.name
                    source.charBegin := token.stringStart
                    source.charEnd := token.stringEnd
                    source.tokenBegin := token.position
                    source.tokenEnd := token.position
                    source.phrase := token.string
                    source.canonical := entstr
                    if(!mentionIndexMap.contains(source.charBegin()+"-"+source.charEnd())) {
                      mentionSeq += source
                      mentionIndexMap += (source.charBegin()+"-"+source.charEnd() ) -> source
                    }  else{
                      val origMention = mentionIndexMap(source.charBegin()+"-"+source.charEnd())
                      origMention.canonical := entstr
                    }
                  }
                }
              }
            }
          }
        //}
      }

      for(sentence <- document.sentences){
        //add common noun and pronoun mentions
        val tokens = sentence.tokens
        for (index <- 0 until tokens.length) {
          val token = tokens(index)
          if(token.attr[PosTag].value.compare( "PRP") == 0 || token.attr[PosTag].value.compare("PRP$") == 0 ) {
            val mention = new Mention
            mention.docId := document.name
            mention.tokenBegin := token.position
            mention.tokenEnd := token.position
            mention.charBegin := token.stringStart
            mention.charEnd := token.stringEnd
            mention.phrase := document.string.substring(mention.charBegin(),mention.charEnd())
            mention.canonical := "None"
            if(!mentionIndexMap.contains((mention.charBegin()+"-"+mention.charEnd()))) {
              mentionSeq += mention
              mentionIndexMap += (mention.charBegin() + "-"+ mention.charEnd()) -> mention
        //      println("Add mention:" + mention.phrase())
            } else{
        //      println("Skip mention:" + mention.phrase())
            }
          }

         /* if (token.attr[NerTag].value.compare("O") == 0 && token.attr[PosTag].value.startsWith("N")) {
            val mention = new Mention
            mention.docId := document.name
            mention.tokenBegin := token.position
            mention.tokenEnd := token.position
            mention.charBegin := token.stringStart
            mention.charEnd := token.stringEnd
            mention.phrase := document.string.substring(mention.charBegin(),mention.charEnd())
            mention.canonical := "None"
            if(!mentionIndexMap.contains((mention.charBegin()+"-"+mention.charEnd()))) {
              mentionSeq += mention
              matchedMentions += mention.charBegin() -> mention.charEnd()
            }
          }*/
        }
      }

      if(mentionSeq.size < 300){
        val predMap = Coref.processDocument(document,mentionSeq,featureStyle,clsf,scorer.pairwiseClassification, canonicals)
   //     println(predMap)

      }else if(mentionSeq.filter(x => x.canonical().compare("None")!= 0).size < 500){
        val predMap = Coref.processDocument(document,mentionSeq.filter(x => x.canonical().compare("None")!= 0),featureStyle,clsf,scorer.pairwiseClassification, canonicals)
  //      println(predMap)
      }

      for (mention <- mentionSeq; if(mention.canonical().compare("None")!= 0) ) { //for common nouns and pronouns, we only consider those with coreferred entities
        var phrase = mention.canonical()
        //link phrase to tac queries, only deal with prefix, suffix
        if(!canonicals.contains(phrase)){
          for(entstr : String <- canonicals){
            if(phrase.toLowerCase.matches(SUFFIX) || phrase.toLowerCase.matches(PREFIX)) {
              if(phrase.contains(entstr)) phrase = entstr
            }
          }
        }
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
      count += 1
      if(count % 1000 == 0){
        println("Processed documents: " + count)
      }
    }
  }

}

