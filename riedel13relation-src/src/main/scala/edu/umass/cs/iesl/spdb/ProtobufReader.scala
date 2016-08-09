package edu.umass.cs.iesl.spdb

import cc.factorie.app.nlp._
import cc.factorie.app.strings._
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
import protobuf.DocumentProtos
import collection.mutable


/**
 * @author lmyao
 * read in protobuf files, get the cells out
 */
object ProtobufReader extends HasLogger {


  def main(args: Array[String]) {
    //create factorie NLP objects here
    val dir = if(args.length > 0) args(0) else "/Users/lmyao/Documents/ie_ner/unsuprel/tackbp/LTW_ENG_20070309.0062.LDC2009T13.sgm" //"/iesl/canvas/dietz/tac/TAC_2010_KBP_Source_Data/data"
    val output = if(args.length > 1) args(1) else "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp/docs.triples.dat"
    val vfile = "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp/visualization.txt"
    val docidfile = "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp/docid_orig.txt"


    //process 2010 data
    groupByPairs2010("/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/data-2010/relation-2010-train.triples.universal.txt",
      "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/data-2010/relation-2010-train.triples.universal.ds.txt",
      "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/data-2010/relation-2010-train.triples.universal.mention.txt")

    groupByPairs2010("/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/data-2010/relation-2010-test.triples.universal.txt",
      "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/data-2010/relation-2010-test.triples.universal.ds.txt",
      "/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/data-2010/relation-2010-test.triples.universal.mention.txt")

    val configFile =  "tackbp-lmyao-1.conf"
    Conf.add(configFile)
 //   val allowed = Util.loadHashSet("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/patterns.txt").toSet
//    Entity.groupByEntity("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/cand.unary.uw.txt.dat", allowed)
 //   Entity.refilter("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/cand.unary.uw.txt.dat.attr", allowed)
   // Entity.extractFreebaseTypes("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/cand.unary.txt.dat.attr", "/iesl/canvas/lmyao/ie_ner/relation/freebase/dump-14-Oct-2010/nyt-freebase-simple-topic-dump-3cols.tsv",
   //   "/iesl/canvas/lmyao/ie_ner/unsuprel/nyt-freebase-simple-topic-20years.tsv")    
  //  Entity.filterFrTypes("/iesl/canvas/lmyao/ie_ner/unsuprel/nyt-freebase-simple-topic-20years.tsv")
  //  Entity.filterByUWTypes("/iesl/canvas/lmyao/ie_ner/unsuprel/nyt-freebase-simple-topic-20years.tsv", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/fr2types_uw.map") //filter by uw types

  //  Entity.matchingEntitiesWithFreebase("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/cand.unary.uw.txt.dat.attr", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/nyt-freebase-simple-topic-20years.labeled.tsv")
  //  Entity.matchingEntitiesWithFreebase("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/exp.txt.unary.txt.dat.attr", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/exp.gold.label")
    
//    Entity.prepareME("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/cand.unary.train.txt.fr.labeled", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/cand.unary.train.binary.ds.txt")
//    Entity.prepareME("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/cand.unary.test.txt.fr.labeled", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/cand.unary.test.binary.ds.txt", true)

 //   Entity.getRank("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/binary/cand.unary.test.pred.txt", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/cand.unary.test.txt.fr.labeled")

//    Entity.getRecPrec("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/binary/cand.unary.test.pred.txt", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/cand.unary.test.txt.fr.labeled",
//      "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/binary/cand.unary.test.pred.rank.txt")

//    Entity.getRecPrec("/iesl/canvas/lmyao/workspace/spdb/out/2_3_2013/run_11_45_58_652/cand.unary.test.rank.txt", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/exp.gold.label",
//      "/iesl/canvas/lmyao/workspace/spdb/out/2_3_2013/run_11_45_58_652/cand.unary.test.rank.txt")
 //   Entity.evalByType("/iesl/canvas/lmyao/workspace/spdb/out/2_3_2013/run_16_11_25_448/cand.unary.test.rank.txt", "/iesl/canvas/lmyao/ie_ner/unsuprel/unary/uw_figer/exp.gold.label", 0.13787710872629944)

//    Entity.getRankByType("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/me.binary.rank.txt")
//    Entity.getRankByType("/iesl/canvas/lmyao/ie_ner/unsuprel/unary/pca.rank.txt")

    //    fromProtobufFile(new File(Conf.conf.getString("relation-data.heldoutDir"),"trainPositive.pb"),
    //      new File(Conf.conf.getString("relation-data.heldoutDir"),"trainNegative.pb"),
    //      Conf.conf.getString("relation-data.heldoutDir")+"/relation-2010-train")
    //
    //    //test is not used at all during training, this only performs as ground truth
    //    fromProtobufFile(new File(Conf.conf.getString("relation-data.heldoutDir"),"testPositive.pb"),
    //      new File(Conf.conf.getString("relation-data.heldoutDir"),"testNegative.pb"),
    //      Conf.conf.getString("relation-data.heldoutDir")+"/relation-2010-test")

    //    Entity.fromProtobufFile(new File(Conf.conf.getString("relation-data.heldoutDir"),"entities.pb"),
    //      Conf.conf.getString("relation-data.heldoutDir")+"/entity-2010" )
    //
    //    produceTriples(Conf.conf.getString("relation-data.heldoutDir")+"/relation-2010-train.triples.guid.txt" ,
    //      Conf.conf.getString("relation-data.heldoutDir")+"/entity-2010.txt",
    //      Conf.conf.getString("relation-data.heldoutDir")+"/relation-2010-train.triples.ds.txt"
    //    )
    //
    //    produceTriples(Conf.conf.getString("relation-data.heldoutDir")+"/relation-2010-test.triples.guid.txt" ,
    //      Conf.conf.getString("relation-data.heldoutDir")+"/entity-2010.txt",
    //      Conf.conf.getString("relation-data.heldoutDir")+"/relation-2010-test.triples.ds.txt"
    //    )
    val heldoutDir =  Conf.conf.getString("relation-data.heldoutDir")
//    var filter = Util.loadHashSet(heldoutDir + "/nyt-freebase.pairs.txt") //all freebase pairs
//    extractTriples("/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/candidate-nyt-20years.filtered.dat",
//      Conf.conf.getString("relation-data.heldoutDir")+"/nyt-freebase.train.triples.txt",
//      Conf.conf.getString("relation-data.heldoutDir")+"/nyt-freebase-entities.txt",
//      Conf.conf.getString("relation-data.heldoutDir")+"/nyt-freebase.train.triples.universal.mention.txt",
//      "200[01234567]", filter)
//    groupByPairs(heldoutDir+"/nyt-freebase.train.triples.universal.mention.txt", heldoutDir+"/nyt-freebase.train.triples.universal.txt")

//    var filter = Util.loadHashSet(heldoutDir+"/nyt-freebase.train.pairs.txt")   //freebase pairs + training pairs
//    extractTriples("/iesl/canvas/lmyao/ie_ner/unsuprel/univSchema/candidate-nyt-20years.filtered.dat",
//      Conf.conf.getString("relation-data.heldoutDir")+"/nyt-freebase.test.triples.txt",
//      Conf.conf.getString("relation-data.heldoutDir")+"/nyt-freebase-entities.txt",
//      Conf.conf.getString("relation-data.heldoutDir")+"/nyt-freebase.test.triples.universal.mention.txt",
//      "199[0123456789]", filter)
//
//    groupByPairs(heldoutDir+"/nyt-freebase.test.triples.universal.mention.txt", heldoutDir+"/nyt-freebase.test.triples.universal.txt")

//   prepareData4mihai(heldoutDir+"/nyt-freebase.train.triples.universal.mention.txt", heldoutDir+"/ds/patterns.txt", heldoutDir+"/ds/trainTuples.txt", heldoutDir+"/ds/labels.txt", heldoutDir+"/mihai/nyt-freebase.train.universal.miml.txt")
//   //prepareData4mihai(heldoutDir+"/nyt-freebase.test.triples.universal.mention.txt", heldoutDir+"/ds/patterns.txt", heldoutDir+"/ds/testTuples.txt", heldoutDir+"/ds/labels.txt", heldoutDir+"/mihai/nyt-freebase.test.universal.miml.txt")
//   prepareData4mihai(heldoutDir+"/nyt-freebase.test.triples.universal.mention.txt", heldoutDir+"/ds/patterns.txt", heldoutDir+"/ds/devTuples.txt", heldoutDir+"/ds/labels.txt", heldoutDir+"/mihai/nyt-freebase.dev.universal.miml.txt")

    //filterData(heldoutDir+"/nyt-freebase.train.triples.universal.txt", heldoutDir+"/ds/patterns.txt", heldoutDir+"/ds/trainTuples.txt", heldoutDir+"/ds/labels.txt", heldoutDir+"/ds/nyt-freebase.train.universal.ds.txt")
    //filterData(heldoutDir+"/nyt-freebase.test.triples.universal.txt", heldoutDir+"/ds/patterns.txt", heldoutDir+"/ds/testTuples.txt", heldoutDir+"/ds/labels.txt", heldoutDir+"/ds/nyt-freebase.test.universal.ds.txt")
    //filterData(heldoutDir+"/nyt-freebase.test.triples.universal.txt", heldoutDir+"/ds/patterns.txt", heldoutDir+"/ds/devTuples.txt", heldoutDir+"/ds/labels.txt", heldoutDir+"/ds/nyt-freebase.dev.universal.ds.txt")

 //   processME(heldoutDir+"/ds/nyt-freebase.dev.universal.pred.txt", heldoutDir+"/nyt-freebase.test.triples.txt", 2765, heldoutDir+"/ds/nyt-freebase.dev.universal.pred.txt.curve" )

//    val mihaiDir = heldoutDir +  "/mihai/mimlre-2012-06-13/corpora/multir/4_12_2012"
//    processResult4mihai(mihaiDir+"/dev.tuples.pred.txt", mihaiDir+"/dev.tuples", mihaiDir+"/dev.tuples.miml.pred.txt")

    //output format:  docid  entity-pair-str  feature feature (including src/dst as features)
    //prepareData4unsuprel(heldoutDir+"/nyt-freebase.test.triples.universal.mention.txt", heldoutDir+"/rel-lda/nyt-freebase.test.triples.universal.mention.fea.txt")
//    extractTopicFeatures(heldoutDir+"/rel-lda/nyt-freebase.all.triples.universal.mention.fea.assign", heldoutDir+"/rel-lda/nyt-freebase.all.triples.universal.topic.txt")
//    addTopicFeatures(heldoutDir+"/ds/nyt-freebase.train.universal.ds.txt", heldoutDir+"/rel-lda/nyt-freebase.all.triples.universal.topic.txt", heldoutDir+"/ds/nyt-freebase.train.universal.ds.topic.txt")
//    addTopicFeatures(heldoutDir+"/ds/nyt-freebase.dev.universal.ds.txt", heldoutDir+"/rel-lda/nyt-freebase.all.triples.universal.topic.txt", heldoutDir+"/ds/nyt-freebase.dev.universal.ds.topic.txt")
//   if( Util.checkOverlap(new File(heldoutDir+"/ds/tmp.txt"), Util.loadHashSet(heldoutDir+"/ds/testTuples.txt")) )
//     println("Overlapping train and test")
//    else println("No overlapping train and test")
  }

  def produceTriples(relfile: String,  entfile : String, triplefile : String){
    val entMap = Util.loadHash(entfile)
    val os = new PrintStream(triplefile)
    val source = Source.fromFile(relfile)
    for (line <- source.getLines) {
      val fields = line.split("\t")
      val src = fields(0)
      val dst = fields(1)
      if(entMap.contains(src) && entMap.contains(dst))  {
        os.println(fields(2) + "\t" + entMap(src) + "\t" + entMap(dst))
      }  else logger.info("Missing " + line)
    }
    os.close
  }

  //adapted from Relation.scala in refectorie/relation project
  def fromProtobufInputStream(in: InputStream,  os : PrintStream, tos : PrintStream) : DocumentProtos.Relation = {     //todo:check type

    val msg : DocumentProtos.Relation = DocumentProtos.Relation.parseDelimitedFrom(in)
    if (msg == null) return null

    val srcid = msg.getSourceGuid
    val destid = msg.getDestGuid
    val relType = msg.getRelType
    os.print(srcid + "\t" + destid + "\tREL$" + relType  )      //to distinguish relation labels from other features
    tos.println(srcid + "\t" + destid) //todo: get entity name

    for (index <- 0 until msg.getMentionCount) {
      val mentionMsg = msg.getMention(index)
      for(index <- 0 until mentionMsg.getFeatureCount) os.print("\t" + mentionMsg.getFeature(index))
      val sentence = mentionMsg.getSentence
      tos.println(sentence)
    }
    os.println
    msg
  }

  def fromProtobufFile(posfile: File, negfile : File, outPrefix : String) {
    var in = new FileInputStream(posfile)

    val os = new PrintStream(outPrefix + ".triples.txt")
    val tos = new PrintStream(outPrefix + ".orig.txt")        //pair, sentence
    var msg : DocumentProtos.Relation = null

    do {
      msg = fromProtobufInputStream(in, os, tos) //load relation and its mentions with features
    } while (msg != null)
    in.close
    in = new FileInputStream(negfile)
    do {
      msg = fromProtobufInputStream(in, os, tos) //load relation and its mentions with features

    } while (msg != null)
    os.close
    tos.close
    in.close
  }

  object Entity {

    def fromProtobufFile(file : File, outPrefix : String){
      var msg : DocumentProtos.Entity = null
      val in = new FileInputStream(file)
      val os = new PrintStream(outPrefix + ".txt")
      do {
        msg = fromProtobufInputStream(in, os) //load relation and its mentions with features

      } while (msg != null)
      os.close
      in.close
    }

    def fromProtobufInputStream(in: InputStream, os : PrintStream) : DocumentProtos.Entity =  {
      val msg = DocumentProtos.Entity.parseDelimitedFrom(in)
      if (msg == null) return null
      val guid = msg.getGuid
      val name = msg.getName
      val types = msg.getType

      os.println(guid + "\t" + name + "\t" + types)

      for (index <- 0 until msg.getMentionCount) {
        msg.getMention(index)
      }
      msg
    }
    
    // prepare entity-attr matrix, line: entity string
    def groupByEntity(file : String, extraRelations: Set[String] = Set.empty) {
      val threshold = 200
      val pat2freq : HashMap[String, Int] = new HashMap[String, Int]
      val ent2attr = new HashMap[String,HashMap[String,Int]]
      val source = Source.fromFile(file)
      for (line <- source.getLines(); if(!line.startsWith("#Document")) ){
        val fields = line.split("\t")
        val attrMap = ent2attr.getOrElseUpdate(fields(0), new HashMap[String, Int])
        
        for (attr <- fields.drop(2); if((!attr.startsWith("sen#") && !attr.contains ("say") && !attr.contains( "-be-") && !attr.endsWith("-be") && attr != "->nn->--------------------") ) || extraRelations.contains(attr) ) {
          if (!attrMap.contains(attr)) attrMap += attr -> 0
          attrMap.update(attr,attrMap(attr)+1)

          if(!pat2freq.contains(attr)) pat2freq += attr -> 0
          pat2freq.update(attr, pat2freq(attr)+1)
        }
      }
      
      var os = new PrintStream(file + ".stat")
      os.println(pat2freq.toList.sortWith((x,y)=> x._2 > y._2).map(x => x._1 + "\t" + x._2).mkString("\n")  )
      os.close()
      
      os = new PrintStream(file + ".attr")
      for ( (ent, attrMap) <- ent2attr ){
        val attrs = attrMap.toList.filter(x=> ( (pat2freq(x._1)>threshold && x._2 > 1 ) || extraRelations.contains (x._1) )  ).sortBy(-_._2).take(50).map(_._1).mkString("\t")

        if (attrs.length > 0) os.println(ent + "\t" + attrs )
      }
      os.close
    }

    def refilter(file : String, extraRelations: Set[String] = Set.empty){
      val threshold = 10
      val pat2freq : HashMap[String, Int] = new HashMap[String, Int]

      val source = Source.fromFile(file)
      for (line <- source.getLines() ){
        val fields = line.split("\t")

        for (attr <- fields.drop(1); if( ( !attr.contains ("say") && !attr.contains( "-be-") && !attr.endsWith("-be") && attr != "->nn->--------------------")  || extraRelations.contains(attr) ) ) {

          if(!pat2freq.contains(attr)) pat2freq += attr -> 0
          pat2freq.update(attr, pat2freq(attr)+1)
        }
      }



      val os = new PrintStream(file + ".filtered")
      for ( line <- Source.fromFile(file).getLines() ){
        val fields = line.split("\t")
        val attrs = fields.drop(1).filter(x=> ( pat2freq(x)>threshold  || extraRelations.contains (x) )  ).mkString("\t")

        if (attrs.length > 0) os.println(fields(0) + "\t" + attrs )
      }
      os.close
    }
    
    // get freebase entity types 
    def extractFreebaseTypes(attrFile : String, freebasefile : String, output : String){
      val os = new PrintStream(output)
      val entities = Util.loadHash(attrFile)
      val source = Source.fromFile(freebasefile)
      for (line <- source.getLines()) {
        val fields = line.split("\t")
        if (entities.contains(fields(1))){
          os.println(fields.drop(1).mkString("\t"))
        }
      }
      os.close
    }
    
    //filter freebase types, too frequent ones and low frequent ones
    def filterFrTypes(input : String){
      val threshold = 100
      val idfThreshold = 0.8
      val type2freq : HashMap[String, Int] = new HashMap[String, Int]
      var source = Source.fromFile(input)
      var lineno = 0
      for (line <- source.getLines()){
        val fields = line.split("\t")
        val typeinfo = fields(1)
        val entTypes = typeinfo.split(",")
        for (entType <- entTypes ){
          if(!type2freq.contains(entType)) type2freq += entType -> 0
          type2freq.update(entType, type2freq(entType)+1)
        }
        lineno += 1
      }

      var os = new PrintStream(input + ".stat")
      os.println(type2freq.toList.sortWith((x,y)=> x._2 > y._2).map(x => x._1 + "\t" + x._2).mkString("\n")  )
      os.close
      
      os = new PrintStream(input + ".filtered")
      source = Source.fromFile(input)
      for (line <- source.getLines()){
        val fields = line.split("\t")
        val entTypes = new ArrayBuffer[String]
        for (entType <- fields(1).split(",")){
          val freq = type2freq(entType)
          if (freq > threshold && freq*1.0/lineno < idfThreshold )  entTypes += "REL$" + entType
        }
        if (entTypes.size > 0) os.println(fields(0) + "\t" + entTypes.mkString("\t") )
      }
      os.close
    }
    
    def filterByUWTypes(input : String,  labelfile : String) {
      val labels = Util.loadHash(labelfile)

      val os = new PrintStream(input + ".filtered")

      val source = Source.fromFile(input)
      for (line <- source.getLines()){
        val fields = line.split("\t")
        val entTypes = new HashSet[String]
        for (entType <- fields(1).split(",")){
          if (labels.contains(entType) )  entTypes += "REL$" + labels(entType)
        }
        if (entTypes.size > 0) os.println(fields(0) + "\t" + entTypes.mkString("\t") )
      }
      os.close
    }
    
    def matchingEntitiesWithFreebase(file : String, labelfile : String){
      val labelsMap = new HashMap[String,String]
      for (line <- Source.fromFile(labelfile).getLines()){
        val fields = line.split("\t")
        labelsMap += fields(0) -> fields.drop (1).mkString ("\t")
      }
      
      val os = new PrintStream(file+".fr.labeled")
      for (line <- Source.fromFile(file).getLines()) {
        val fields = line.split("\t")
        if (labelsMap.contains(fields(0)))
          os.println(line+"\t"+labelsMap(fields(0)))
        else os.println(line)
      }
      os.close
    }
    
    def prepareME(file : String, mefile : String, test : Boolean = false){
      val os = new PrintStream(mefile)
      for (line <- Source.fromFile(file).getLines(); if (line.contains("REL$"))) {
        val fields = line.split("\t")
        val features = new ArrayBuffer[String]()
        val labels = new ArrayBuffer[String]()
        for (field <- fields.drop(1)){
          if (field.startsWith("REL$")) labels += field
          else features += field
        }
        if (!test) {
//          for (label <- labels){
//            os.println (label.replaceAll ("REL\\$","") + "\t" + fields(0) + "\t" + features.mkString("\t") )
//          }
          os.println(labels.map(_.replaceAll("REL\\$","")).mkString(",") + "\t" + fields(0) + "\t" + features.mkString("\t"))
        }
        else os.println(labels(0).replaceAll ("REL\\$","") + "\t" + fields(0) + "\t" + features.mkString("\t") )
      }
      os.close
    }
    
    def getRank(predfile : String, labelfile : String){
      val goldInstances = new ArrayBuffer[(String, Set[String])]
      val ranks = new ArrayBuffer[Int]
      for (line <- Source.fromFile(labelfile).getLines(); if (line.contains("REL$"))) {
        val fields = line.split("\t")
        val labels = fields.drop(1).filter(_.startsWith("REL$")).map(_.replaceAll("REL\\$","")).toSet
        goldInstances += fields(0) -> labels      //array buffer, each cell is entity->labels
      }
      val os = new PrintStream(predfile + ".rank")
      var id = 0
      for  (line <- Source.fromFile(predfile).getLines()) {
        val fields = line.split("\t")
        val labels = goldInstances(id)._2
        
        var rank = 1
        os.println(fields(0))

        //unsorted label:score label:score
        val sorted = fields.drop(1).sortWith((x,y)=> {val xscore = x.split(":")(1).toDouble; val yscore = y.split(":")(1).toDouble; xscore > yscore})
        for (field <- sorted){
        //sorted labels
        //for (field <- fields.drop(2)){
          val label = field.split(":") (0)
          val score = field.split(":")(1).toDouble
          if (labels.contains(label)) {
            os.println("Rank\t" + fields(0) + "\t" + label + "\t" + score + "\t" + rank)
            ranks += rank
          }
          else if( rank <= 10 ) os.println(fields(0) + "\t" + label + "\t" + score)
          rank += 1
        }
        
        id += 1
      }

      val sum = ranks.foldLeft(0.0)((x,y) => x + y)
      os.println("Size:" + ranks.size)
      os.println("Mean rank:" + sum*1.0/ranks.size)
      val sortedRanks = ranks.sortWith((x,y)=>x<y)
      os.println("Median rank:" + sortedRanks(ranks.size/2))
      os.close
    }
    
    def getRecPrec(predfile : String,  labelfile : String, output : String){
      val goldInstances = new HashMap[String,HashSet[String]]  // entity->label
      var gold = 0
      for (line <- Source.fromFile(labelfile).getLines(); if (line.contains("REL$"))) {
        val fields = line.split("\t")
        val labels = fields.drop(1).filter(_.startsWith("REL$")).map(_.replaceAll("REL\\$","")).toSet
        for (label <- labels)
          goldInstances.getOrElseUpdate (fields(0), new HashSet[String]) += label
        gold += labels.size
      }

   /*   val preds = new ArrayBuffer[(Double, String, String, String)]
      for  (line <- Source.fromFile(predfile).getLines()) {
        val fields = line.split("\t")
        val goldlabel = goldInstances(fields(0)).toSeq(0)
       
        for (field <- fields.drop(1)){
          val label = field.split(":") (0).replaceAll("REL\\$","")
          val score = field.split(":")(1).toDouble
          preds += ( (score, fields(0), goldlabel,  label) )
        }
      }

      val sorted = preds.sortBy(-_._1)   */

      val sorted = new ArrayBuffer[(Double, String, String, String)]
      for  (line <- Source.fromFile(predfile).getLines()) {
        val fields = line.split("\t")

        sorted += ( (fields(0).toDouble, fields(1), fields(2).replaceAll("REL\\$",""),  fields(3).replaceAll("REL\\$","") ) )

      }

      var cor = 0
      var total = 0
      val scores = new ArrayBuffer[(Double,Double, Double)]
    //  val os = new PrintStream(output)
      val ocurve = new PrintStream(output+".curve")

      for ((score, entity, goldLabel, label) <- sorted) {

        if (goldInstances(entity).contains(label)) {
          cor += 1
        }
        total += 1
       // if (total % 100 == 0)
        val rec = cor * 1.0 / gold
        val prec = cor * 1.0 / total
        val f1 = 2 * prec * rec / (prec+rec)
        ocurve.println(total + "\t" + rec + "\t" + prec + "\t" + f1)
        scores += ((rec,prec,f1))
    //    os.println(score + "\t" + entity + "\t" + goldLabel + "\t" + label)
      }
   //   os.close()
      ocurve.close
      
      println(scores.sortBy(-_._3).take(1).map(score=> score._1 + "\t" + score._2 + "\t" + score._3).mkString("\t"))
    }

    case class Measure() {
      var cor : Int = 0
      var gold : Int = 0
      var total : Int = 0
      var f1max : Double = 0.0
      var auc : Double = 0.0
      var f1maxScores : String = ""
    }
    
    def evalByType(rankfile : String,  labelfile : String, threshold : Double = 0.5) {
      val goldInstances = new HashMap[String,HashSet[String]]  // entity->label
      val numbers = new HashMap[String, Measure]
      for (line <- Source.fromFile(labelfile).getLines(); if (line.contains("REL$"))) {
        val fields = line.split("\t")
        val labels = fields.drop(1).filter(_.startsWith("REL$")).map(_.replaceAll("REL\\$","")).toSet
        for (label <- labels) {
          goldInstances.getOrElseUpdate (fields(0), new HashSet[String]) += label
          numbers.getOrElseUpdate(label, new Measure).gold += 1
        }
      }

      val sorted = new ArrayBuffer[(Double, String, String, String)]
      for  (line <- Source.fromFile(rankfile).getLines()) {
        val fields = line.split("\t")
        val score = fields(0).toDouble
        //if (score > threshold)
          sorted += ( (score, fields(1), fields(2).replaceAll("REL\\$",""),  fields(3).replaceAll("REL\\$","")) )

      }

   
      val os = new PrintStream(rankfile +".eval")

      for ((score, entity, goldLabel, label) <- sorted) {

        if (goldInstances(entity).contains(label)) {
          numbers.getOrElseUpdate(label, new Measure).cor += 1
        }
        numbers.getOrElseUpdate(label, new Measure).total += 1
        val prec = numbers(label).cor * 1.0 / numbers(label).total
        val rec = if (numbers(label).gold>0) numbers(label).cor * 1.0 / numbers(label).gold  else 0.0
        numbers.getOrElseUpdate(label, new Measure).auc +=  prec
        val f1 =  2 * prec * rec / (prec + rec)
        if (f1 > numbers(label).f1max) {
          numbers(label).f1max = f1
          numbers(label).f1maxScores = numbers(label).cor + "\t" +  numbers(label).gold + "\t" + numbers(label).total
        }
      }

      var cor = 0
      var gold = 0
      var total = 0
      
      for ((rel,relNumbers) <- numbers){
        val auc = relNumbers.auc / relNumbers.total
        if (relNumbers.f1maxScores.length() < 1)  relNumbers.f1maxScores = relNumbers.cor + "\t" + relNumbers.gold + "\t" + 0
        os.println(rel + "\t" + relNumbers.f1maxScores +  "\t"  + relNumbers.f1max + "\t" + auc)
        val fields = relNumbers.f1maxScores.split ("\t")
        cor += fields(0).toInt
        gold += fields(1).toInt
        total += fields(2).toInt
      }
      val prec = cor * 1.0 / total
      val rec = cor * 1.0 / gold 
      val f1 = 2 * prec * rec / (prec + rec)
      os.println("Overall\t" + cor + "\t" + gold + "\t" + total + "\t" + rec + "\t" + prec + "\t" + f1)
      os.close()
      
    }
    
    def getRankByType(file : String){
      
      val stat = new HashMap[String, ArrayBuffer[Int]]
      for  (line <- Source.fromFile(file).getLines()) {
        val fields = line.split("\t")
        val label = fields(2)
        val rank = fields(4).toInt
        stat.getOrElseUpdate(label,new ArrayBuffer[Int]) += rank
      }

      val os = new PrintStream(file+".type.info")
      for ((label, ranks) <- stat){
        val sum = ranks.foldLeft(0.0)((x,y) => x + y)
        val sortedRanks = ranks.sortWith((x,y)=>x<y)
        os.println(label + "\t" + sum*1.0/ranks.size + "\t" + sortedRanks(ranks.size/2) + "\t" + ranks.size)
      }
      os.close
    }

  }

  //extract relation 2010 triples from candidate-nyt-20years.dat   , allow is year
  //triple file is  rel arg1 arg2, we need to extract features for these triples from datafile , this is only for test now
  def extractTriples(datafile : String, triplefile:String, entityfile : String, output : String, allow : String = "200[01234567]", filter:HashSet[String] = null) {

    val triples : HashMap[String,String] = Util.loadTripleHash1(triplefile) //   arg1 arg2  relation relation relation
    val entities : HashSet[String] = Util.loadHashSet(entityfile)
    val matched = new mutable.HashSet[String]
    val os = new PrintStream(output)
    val fstream = new FileInputStream(datafile)
    val in = new DataInputStream(fstream)
    val br = new BufferedReader(new InputStreamReader(in))
    var strLine : String = ""
    var docid : String = ""
    strLine = br.readLine
    var relLabel : String = ""
    do {
      if (strLine.startsWith("#Document")){
        docid = strLine.replaceAll("#Document ", "")
        os.println(strLine)
      }else{  //get triples that we are interested
        if (docid.matches(".*\\/"+allow+"\\/.*"))  {
          val fields = strLine.split("\t")
          val triggerinfo = fields(0).replaceAll("trigger#", "")
          val triggers = triggerinfo.split(",")
          val src = fields(1)
          val dst = fields(2)

          var srcnerinfo : String = ""
          var dstnerinfo : String = ""
          if (fields(3).split(" ")(0).split("\\/").length < 3){
            logger.info(fields(3))
          } else
            srcnerinfo = fields(3).split(" ")(0).split("\\/")(2)
          if (fields(4).split(" ")(0).split("\\/").length < 3){
            logger.info(fields(4))
          } else
            dstnerinfo = fields(4).split(" ")(0).split("\\/")(2)

            val otherfeatures = fields.drop(7).filter(x => !x.startsWith("lex#"))
            var lexfea = fields.drop(7).filter(_.startsWith("lex#")).mkString("")
            val cleaned =  cleanLex(lexfea.replaceAll("lex#",""))
            if (cleaned.length > 0)
              lexfea = "lex#" + cleaned
            else lexfea = ""

          if (triples.contains(src+"\t"+dst)){
            matched += src + "\t" + dst
            relLabel = triples(src+"\t"+dst).split("\t").map (l => "REL$" + l).mkString ("\t")

            // trigger src dst src/ner dst/ner path sen  other features
            os.print("POSITIVE\t" + src + "\t" + dst + "\t" + relLabel + "\tner#"+srcnerinfo +"->"+dstnerinfo + "\t" + fields(5)+"\t")
            if (lexfea.length>0)
              os.println(otherfeatures.mkString("\t") + "\t" + lexfea + "\t" + triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
            else
              os.println(otherfeatures.mkString("\t")  + "\t" + triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
          } else if (triples.contains(dst + "\t" + src)){
            matched += dst + "\t" + src
            relLabel = triples(dst+"\t"+src).split("\t").map (l => "REL$" + l).mkString ("\t")
            //if (!relLabel.startsWith("REL"))    relLabel = "REL$" + relLabel
            os.print("POSITIVE\t" + dst + "\t" + src + "\t" + relLabel + "\tner#"+dstnerinfo +"->"+srcnerinfo + "\t" + fields(5)+":INV\t")
            if (lexfea.length>0)
              os.println(otherfeatures.map(x =>  x + ":INV").mkString("\t") + "\t" + lexfea+":INV" + "\t" +triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
            else
              os.println(otherfeatures.map(x =>  x + ":INV").mkString("\t") + "\t" +triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
          }else if ((filter != null && !filter.contains(src+"\t"+dst) && !filter.contains(dst+"\t"+src)) ){ // not included in any training triples
            //check whether two args are in Freebase entities, if so, create negative example,
            relLabel = "REL$NA"
            if (entities.contains(src) && entities.contains(dst)){
              os.print("NEGATIVE\t" + src + "\t" + dst + "\t" + relLabel + "\tner#"+srcnerinfo +"->"+dstnerinfo + "\t" + fields(5)+"\t")
              if (lexfea.length > 0)
                os.println(otherfeatures.mkString("\t") + "\t" + lexfea + "\t" + triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
              else
                os.println(otherfeatures.mkString("\t")  + "\t" + triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
            }else{            //if not in Freebase, create unlabeled example
              os.print("UNLABELED\t" + src + "\t" + dst + "\tner#"+srcnerinfo +"->"+dstnerinfo + "\t" + fields(5)+"\t")
//              relLabel = "REL$NA"
//              os.print("NEGATIVE\t" + src + "\t" + dst + "\t" + relLabel + "\tner#"+srcnerinfo +"->"+dstnerinfo + "\t" + fields(5)+"\t")
              if (lexfea.length > 0)
                os.println(otherfeatures.mkString("\t") + "\t" + lexfea + "\t" + triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
              else
                os.println(otherfeatures.mkString("\t")  + "\t" + triggers.map(x=>"trigger#"+x).mkString("\t") + "\t" + fields(6))
            }
          }

        }
      }
      strLine = br.readLine
    } while(strLine != null)
    in.close
    os.close

    var count = 0
    for (tuple <- triples.keySet){
      if (!matched.contains(tuple)) count += 1
    }
    logger.info("Missing tuples:" + count)
  }

  def cleanLex(lexfea : String) : String = {
    val words = lexfea.split(" ")
    val usefulwords = new ArrayBuffer[String]
    for (word <- words){
      if (word.length > 1 && !Stopwords.contains(word)) usefulwords += word
    }
    return usefulwords.mkString(" ")
  }

  //./2007/02/20/1827616.xml        ->appos->chairman->prep->of->pobj->|David E. Cole|Center for Automotive Research        ->appos->chairman->prep->of->pobj->#1   l:PERSON#1      r:research#1    l:cole#1        r:ORGANIZATION#1        r:center#1      lex17#1 Topic2#1.5      r:automotive#1  chairman#1      l:david#1
  //  ./2007/02/20/1827616.xml        ->appos->spokesman->prep->for->pobj->|Tom Wilkinson|G.M.        ->appos->spokesman->prep->for->pobj->#1 l:tom#1 l:PERSON#1      r:ORGANIZATION#1        spokesman#1     r:g m#1 lex17#1 Topic2#1.5      l:wilkinson#1

  def prepareData4unsuprel(input : String, output : String){
    val os = new PrintStream(output)
    val fstream = new FileInputStream(input)
    val in = new DataInputStream(fstream)
    val br = new BufferedReader(new InputStreamReader(in))
    var strLine : String = ""
    var docid : String = ""
    strLine = br.readLine

    do {
      if (strLine.startsWith("#Document")){
        docid = strLine.replaceAll("#Document ", "")
      } else if(strLine.startsWith("UNLABELED")){}   //skip unlabeled
      else{
        //process each mention
        val fields = strLine.split("\t").filterNot(_.startsWith("REL$")) //the first is POSITIVE/NEGATIVE/UNLABELED
        val src = fields(1)
        val dst = fields(2)
        val nerInfo = fields(3).replaceAll("ner#","")
        os.print(docid + "\t" + src+"|"+dst + "\t")

        val srcWords = src.split(" ")
        os.print(srcWords.map(_.toLowerCase).map(f => "l:"+f).mkString("\t")+"\t")
        val dstWords = dst.split(" ")
        os.print(dstWords.map(_.toLowerCase).map(f => "r:"+f).mkString("\t")+"\t")
        val splitPos = nerInfo.indexOf("-")
        val srcNer = nerInfo.substring(0,splitPos)
        val dstNer = nerInfo.substring(splitPos+2,nerInfo.length)
        //os.print("l:"+srcNer + "\tr:" + dstNer+"\t")
        //os.println(fields.drop(4).filterNot(f.startsWith("sen#")).mkString("\t"))
        os.println(fields.drop(4).filter(_.startsWith("path#")).mkString("\t"))
      }
      strLine = br.readLine
    }while(strLine != null)
    os.close
    in.close
  }
  
  def extractTopicFeatures(inputfile : String,  outputfile : String){
    val pair2Topic = new HashMap[String,HashSet[String]]
    val source = Source.fromFile(inputfile)
    for(line <- source.getLines() ){
      val fields = line.split("\t")
      if(fields.size > 1){
        val topic = fields(fields.length-1)
        val pair = fields(1)
        pair2Topic.getOrElseUpdate(pair,new HashSet[String]) += topic
      }
    }
    source.close
    
    val os = new PrintStream(outputfile)
    for ((pair,topics) <- pair2Topic) os.println(pair + "\t" + topics.mkString("\t") )
    os.close
  }
  
  def addTopicFeatures(dsfile : String, topicfile: String,  outputfile : String){
    val pair2Topic = new HashMap[String,String]
    var source = Source.fromFile(topicfile)
    for (line <- source.getLines()) {
      val fields = line.split("\t")
      pair2Topic += fields(0) -> fields.drop(1).mkString ("\t")
    }
    source.close()
    
    val os = new PrintStream(outputfile)
    source = Source.fromFile(dsfile)
    for (line <- source.getLines()) {
      val fields = line.split("\t")
      val pair = fields(1)
      os.print(line)
      if(pair2Topic.contains(pair)) os.print("\t" + pair2Topic(pair))
      os.println
    }
    os.close
  }

  def groupByPairs(inputfile : String,  outputfile : String){
    val pair2featureMap = new HashMap[String, HashSet[String]]
    val source = Source.fromFile(inputfile)
    for(line <- source.getLines(); if(!line.startsWith("#Document")) ){          //
      val fields = line.split("\t")
      if (fields.size > 1){

        val tag = fields(0)
        var pair = fields(1) + "\t" + fields(2)

        var features : HashSet[String]  = pair2featureMap.getOrElseUpdate(pair,new HashSet[String])
        features += tag

        for(field <- fields.drop(3); if(!field.startsWith("sen#")) ){
          features += field
        }
      }
    }
    source.close
    val os = new PrintStream(outputfile)
    var tag : String = ""
    for ((pair, features) <- pair2featureMap){
      if(features.contains("POSITIVE"))  tag = "POSITIVE"
      else if(features.contains("NEGATIVE")) tag = "NEGATIVE"
      else tag = "UNLABELED"
      os.println(tag + "\t" + pair + "\t" + features.filter(_.compareTo(tag) != 0).mkString("\t") )
    }
    os.close
  }

  //process data-2010, from mention file to get ds file, miml file
  //miml format:  RELATION $rel  source dest
  //              MENTION $fea $fea $sen \n MENTION
  def groupByPairs2010(inputfile : String,  outputfile : String, moutputfile : String){
    val pair2featureMap = new HashMap[Seq[Any], HashSet[String]]
    val pair2mention = new HashMap[Seq[Any],ArrayBuffer[String]]
    val source = Source.fromFile(inputfile)
    for(line <- source.getLines() ){          //
      val fields = line.split("\t")
      if (fields.size > 1){

        var pair = Seq(fields(0),  fields(1)  )
        pair2mention.getOrElseUpdate(pair,new ArrayBuffer[String]) += fields.drop (3).mkString ("\t")
        var features : HashSet[String]  = pair2featureMap.getOrElseUpdate(pair,new HashSet[String])

        for(field <- fields.drop(2); if(!field.startsWith("sen#")) ){
          features += field
        }
      }
    }
    source.close
    val os = new PrintStream(outputfile)
    val mos = new PrintStream(moutputfile)

    for ((pair, features) <- pair2featureMap){
      val labels = features.filter(_.startsWith("REL$")).map(_.replaceAll("REL\\$","")).toSet.mkString(",")
      os.println(labels + "\t" + pair.mkString ("|") + "\t" + features.filterNot(_.startsWith("REL$")).mkString("\t") )
      
      mos.println("RELATION\t" + labels + "\t" + pair.mkString("\t") )
      val mentions = pair2mention(pair)
      mos.println(mentions.map(m=> "MENTION\t" + m).mkString("\n"))
    }
    os.close
    mos.close
  }

  //format:   RELATION RELTYPE arg1 arg2 \n MENTION fea fea \t sen, could be adapted from groupByPairs
  //input format:  POSITIVE arg1 arg2 reltype
  def prepareData4mihai(origfile : String,  patternfile : String,  tuplefile : String, labelfile: String, output :String) {
    val patterns : HashSet[String] = Util.loadHashSet(patternfile)
    val tuples : HashSet[String] = Util.loadHashSet(tuplefile)
    val labels : HashSet[String] = Util.loadHashSet(labelfile)
    val pair2mentionMap = new HashMap[String, HashSet[String]]
    val os = new PrintStream(output)
    val source = Source.fromFile(origfile)
    for (line <- source.getLines(); if (!line.startsWith("UNLABELED") && !line.startsWith("#Document"))){
      var fields = line.split("\t")
      if (fields(0) == "POSITIVE" || fields(0) == "NEGATIVE" || fields(0)=="UNLABELED") fields = fields.drop(1)
      val tuple = fields(0) + "|" + fields(1)
      var relType = new ArrayBuffer[String]
      var i = 2
      while (i < fields.length && fields(i).startsWith("REL$")){

        if(labels.contains(fields(i))) relType +=  fields(i).replaceAll ("REL\\$","")
        i += 1

        
      }
      var label = ""
      if (relType.size == 0)  label = "NA"
      else label = relType.mkString(",")

      var mentions : HashSet[String]  = pair2mentionMap.getOrElseUpdate(label+"\t"+tuple,new HashSet[String])
      var features : String = ""
      for (fea <- fields.drop(i); if (patterns.contains(fea) || fea.startsWith("sen#"))) features += fea + "\t"
      mentions += features.substring(0,features.length-1)
    }
    source.close

    for ((triple,mentions) <- pair2mentionMap){
      val tuple = triple.split("\t")(1)
      val label = triple.split("\t")(0)

      if (tuples.contains(tuple)){
        os.println("RELATION\t" + label + "\t" + tuple.split("\\|").mkString("\t"))
        os.println( mentions.map(mention => "MENTION\t" + mention).mkString("\n"))
      }
    }
    os.close
  }

  //tuple file
  //id gold label pred label score label score, more than one labels possible   //test.tuples.pred.txt
  def processResult4mihai(predfile : String, tuplefile : String,  output : String){
    var  source = Source.fromFile(tuplefile)
    val tuples = source.getLines().toSeq
    val preds = new ArrayBuffer[String]
    source = Source.fromFile(predfile)
    
    var gold = 0
    
    val os = new PrintStream(output)
      for (line <- source.getLines()){
      val fields = line.split("\t")
      val index = fields(0).toInt

      os.println(tuples(index) + "\t" + fields.drop(1).mkString ("\t"))


    }
//    val sorted = preds.toList.sortWith((x, y) => {
//      val xscore = x.split("\t")(0).toDouble;
//      val yscore = y.split("\t")(0).toDouble;
//      xscore > yscore
//    })

//    val os = new PrintStream(output)
//    val eos = new PrintStream(output+".curve")
//    var id = 0
//    var cor = 0
//    var total = 0
//
//    for (instance <- sorted ) {
//      val fields = instance.split("\t")
//      if (fields(0) != "0.0") {
//      if (fields(3).compareTo(fields(4)) == 0) {
//        cor += 1
//      }
//      total += 1
//      id += 1
//      eos.println(id + "\t" + cor * 1.0 / gold + "\t" + cor * 1.0 / total)
//      }
//    }
//    os.println(sorted.mkString("\n"))
//    os.close
    //eos.close
  }
  
  //produce the recall-precision curve, using multilabels   goldfile=Conf.conf.getString("relation-data.heldoutDir")+"/nyt-freebase.test.triples.txt"
  def processME(rankfile : String, goldfile : String, gold:Int,   output : String){
    val outputStream = new PrintStream(output )
    val os = new PrintStream(rankfile+".rank")
    var cor = 0
    var total = 0
    var rec = 0.0
    var prec = 0.0
    val source = Source.fromFile(rankfile)
    val goldLabels = Util.loadTripleHash1(goldfile)
    for(predict <- source.getLines()){
      val fields = predict.split("\t")
      val pair = fields(1).split("\\|").mkString("\t")
      os.println(fields(0) + "\t" + pair + "\t" + fields.drop(2).map(label=>"REL$"+label).mkString("\t") )
      val pred = fields(3)
      if (goldLabels.contains(pair) && goldLabels(pair).contains(pred) ) cor += 1
      total += 1
      prec = cor*1.0/total
      rec = cor*1.0/gold
      outputStream.println(total + "\t" + rec + "\t" + prec)
    }
    outputStream.close
    os.close
  }

  def filterData(origfile : String,  patternfile : String,  tuplefile : String, labelfile : String, output :String) {
    val patterns : HashSet[String] = Util.loadHashSet(patternfile)
    val tuples : HashSet[String] = Util.loadHashSet(tuplefile)
    val labels : HashSet[String] = Util.loadHashSet(labelfile)
    
    //for debugging
    val matched = new HashSet[String] 
    
    val os = new PrintStream(output)
    val source = Source.fromFile(origfile)
    for (line <- source.getLines()){
      var fields = line.split("\t")
      if (fields(0) == "POSITIVE" || fields(0) == "NEGATIVE" || fields(0)=="UNLABELED") fields = fields.drop(1)
      val tuple = fields(0) + "|" + fields(1)
      var label = fields.filter(_.startsWith("REL$")).mkString.replaceAll("REL\\$","")
      if (!labels.contains("REL$"+label)){
        //println("rare label:"+ label)
        label = "NA"
      }
      if (tuples.contains(tuple)){
        os.print(label + "\t" + tuple + "\t")
        os.println(fields.drop(2).filter(fea => patterns.contains(fea) && !(fea.startsWith("REL$"))).mkString("\t"))
        //os.println(fields.drop(2).filter(fea => fea.startsWith("path#") && !(fea.startsWith("REL$"))).mkString("\t"))
        matched += tuple
      }
    }
    source.close()
    os.close
    
    for (tuple <- tuples){
      if (!matched.contains(tuple))  println("Missing tuple:" + tuple)
    }
  }


}

