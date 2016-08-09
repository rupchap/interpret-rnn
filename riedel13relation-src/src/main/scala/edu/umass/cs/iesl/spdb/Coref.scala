/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 5/9/12
 * Time: 2:47 PM
 * To change this template use File | Settings | File Templates.
 */
package edu.umass.cs.iesl.spdb
import cc.factorie.app.nlp.coref.PairwiseMention
import cc.refectorie.user.jzheng.coref._
import cc.refectorie.user.sameer.util.coref._

import cc.refectorie.user.jzheng.coref.util.{  MentionUtil, Options }
import collection.mutable.ArrayBuffer
import cc.refectorie.user.jzheng.coref.PosTag
import Conf._
import java.io.{File, PrintStream}
import cc.factorie.app.nlp.{TokenSpan, Document}
import cc.factorie.app.nlp.ner.NerSpan

object Coref {
  
  def main(args : Array[String]){
    
      add("tackbp-lmyao-1.conf")
//    val configFile =  "tackbp-lmyao.conf"
//    Conf.add(configFile)

    val output = "/iesl/canvas/lmyao/ie_ner/unsuprel/tackbp/coref_err.AFP_ENG_20071022.0401.LDC2009T13.txt"
    val os = new PrintStream(output)
    //val os = new PrintStream(new File(Conf.outDir, "visualization.coref.txt"))

    val modelOnDisk = conf.getString("coref.modelOnDisk")//"/iesl/canvas/lmyao/workspace/spdb/duproth_perceptron300_falsefeaturevalues.model"    //on blake,jzheng's dir
    val clsf = Coref.loadClassifier(modelOnDisk, "Perceptron")
    val featureStyle = "DupRoth"
    val scorer = new Scorer[PairwiseMention](Options.predSemEval2010, Options.trueSemEval2010)
    CorefLabel.loadTokenFrequency(conf.getString("coref.tokenFreq"))
    
    for(cd <- Database.CorefDB.documents ;if(cd.name().compareTo("AFP_ENG_20071022.0401.LDC2009T13")==0) ){              //eng-WL-11-174584-12960469   //eng-WL-11-174611-12980242
      val doc = cd.fetchDocument
      println("Processing document " + doc.name)
      val mentionsOfDoc = Database.CorefDB.mentions.query(_.docId(doc.name))
      val mentionSeq = new ArrayBuffer[Mention]
      for(mention <- mentionsOfDoc; if(mention.label() != "NUMBER" && mention.label() != "DATE")){
        mentionSeq += mention
      }
      
      for(sentence <- doc.sentences){
        val tokens = sentence.tokens
        for (index <- 0 until tokens.length) {
          val token = tokens(index)
          if(token.attr[PosTag].value.compare( "PRP") == 0 || token.attr[PosTag].value.compare("PRP$") == 0 ) {
            val mention = new Mention
            mention.docId := doc.name
            mention.tokenBegin := token.position
            mention.tokenEnd := token.position
            mention.charBegin := token.stringStart
            mention.charEnd := token.stringEnd
            mention.phrase := doc.string.substring(mention.charBegin(),mention.charEnd())
            mention.canonical := "None"
            mentionSeq += mention

          }
        }
      }

      val canonicals = new ArrayBuffer[String]
      canonicals += "Ali Fedotowsky"
      canonicals += "New York Mets"
      canonicals += "Mark Buse"
      canonicals += "Wal-Mart"
      val predMap  = processDocument(doc,mentionSeq,featureStyle,clsf,scorer.pairwiseClassification, canonicals)
      
      os.println("Doc:" + doc.name)
      for(sentence <- doc.sentences){
        os.println("Sen:" +sentence.string)
        for(mention <- mentionSeq){
          if(doc.tokens(mention.tokenBegin()).sentence.start == sentence.start){
            os.println("Mention:" + mention.canonical() + "\t" + mention.phrase())
          }
        }
      }

      os.println(predMap)
    }
    os.close
    //postprocess, connect to Tac link
  }
  
  def processDocument(doc: Document, mentionSeq : ArrayBuffer[Mention] , featureStyle: String, classifier: Object, pairwise: BooleanEvaluator, canonicals : Seq[String]) : GenericEntityMap[PairwiseMention] = {

    val pairwiseMents = new ArrayBuffer[TokenSpan]

    for(mention <- mentionSeq){
      val m = new NerSpan(doc, doc(mention.tokenEnd()).attr[NerTag].value, mention.tokenBegin(), mention.tokenEnd()-mention.tokenBegin()+1)(null) with PairwiseMention

      //head is a tokenspan , mId is a uniq id, mType: pronoun, nn, proper noun, eType: org,loc,per,mis
      m.attr += new ACEMentionIdentifiers {
        def mId = "m" + mentionSeq.indexOf(mention)   //todo:check
        def mType = {
          if(doc(mention.tokenEnd()).attr[NerTag].value != "O")  "NAM" //check ACE label
          else if(doc(mention.tokenEnd()).attr[PosTag].value == "PRP" || doc(mention.tokenEnd()).attr[PosTag].value == "PRP$") "PRO"
          else  "NOM"      // "the guy"
        }

        def ldcType =  "LDCTYPE"

        def offsetStart = mention.charBegin()

        def offsetEnd = mention.charEnd()
      }

      m._head = doc(mention.tokenEnd())
      m.attr += new ACEFullHead(new TokenSpan(doc, mention.tokenBegin(), mention.tokenEnd()-mention.tokenBegin()+1))
      pairwiseMents += m

    }
    
    val ments = MentionUtil.getOrderedMentions(doc)

    println("Number of candidates:" + ments.size)
    
    val predMap = new GenericEntityMap[PairwiseMention]
    ments.foreach(m => predMap.addMention(m, predMap.numMentions.toLong))

    for (i <- 0 until ments.size) {
      val m1 = ments(i)
      
     // println("Coref PairwiseMention:" + m1._head.string + "\t" + m1.attr[ACEFullHead].head.phrase.toLowerCase + "\t" + m1.attr[ACEFullHead].head.phrase.trim())

      var bestCand: PairwiseMention = null
      var bestScore = Double.MinValue

      for (j <- Range(i - 1, -1, -1)) {
        val m2 = ments(j)

      //  println("Coref PairwiseMention:" + m2._head.string + "\t" + m2.attr[ACEFullHead].head.phrase.toLowerCase + "\t" + m2.attr[ACEFullHead].head.phrase.trim())

        if (predMap.reverseMap(m1) != predMap.reverseMap(m2)) { // only test when m1 and m2 are not already linked
          val cataphora = m2.attr[ACEMentionIdentifiers].mType.equals("PRO") && !m1.attr[ACEMentionIdentifiers].mType.equals("PRO")
          val trueCoref = true   //this only tells the classifier to calculate p(true|features)

          if (!cataphora || Options.allowTestCataphora) {
            val candLabel = featureStyle match {
              case Options.DupRothFeatures => new DupRothCorefLabel(m1, m2, doc.name, trueCoref)
              case Options.MyFeatures => new MyCorefLabel(m1, m2, doc.name, trueCoref)
              case Options.LbjDumpFeatures => throw new Exception("should use processLbjDump")
              case _ => throw new Exception("unknown featureStyle: " + featureStyle)
            }

            val (score_y, score_n) = UiucCorefTest.scoreOfOnePair(classifier, candLabel) //todo:pay attention to CorefLabelDomain, "YES" and "NO"
            val (predCategory, predCatScore) = if (score_y > score_n) ("YES", score_y) else ("NO", score_n)

            if (predCategory.equals("YES")) {
              val score = score_y
              if (bestScore <= score) {
                bestCand = m2
                bestScore = score

              }
            }

          }
        }

      }

      if (bestCand != null) {
        //println("  adding antecedent: " + bestCand.attr[ACEMentionIdentifiers].mId + " " + bestCand)
        predMap.addCoreferentPair(m1, bestCand)
      }

    }

    //the longest as canonical
    for(entId <- predMap.getEntityIds){
      val mentionsOfEnt = predMap.getMentions(entId).toSeq.sortWith(MentionUtil.beforeInTextualOrder)

      var canonicalMent : PairwiseMention = null //= mentionsOfEnt.head
      var canonicalMention : Mention =null //= mentionSeq(canonicalMent.attr[ACEMentionIdentifiers].mId.replace("m", "").toInt)
      var canonical : String = "" //doc.string.substring ( canonicalMention.charBegin(), canonicalMention.charEnd())

      var matched = false
      var longest : PairwiseMention = null
      var length : Int = 0
      if(mentionsOfEnt.size > 1){
        for(men : PairwiseMention <- mentionsOfEnt) {
          //compare each against tac queries
          val mention = mentionSeq(men.attr[ACEMentionIdentifiers].mId.replace("m", "").toInt)
          val str = mention.phrase()
          if(str.length() > length){
            length = str.length()
            longest = men
          }
//          if(canonicals.contains(str) || (mention.canonical.isDefined && canonicals.contains(mention.canonical()) )){
//            canonicalMent = men
//            canonicalMention = mention
//            canonical = if(canonicals.contains(str))  str else mention.canonical()
//            matched = true
//          }
       //   println("Mention:" + str)
        }

        if(!matched ){
          canonicalMent = longest
          canonicalMention = mentionSeq(canonicalMent.attr[ACEMentionIdentifiers].mId.replace("m", "").toInt)
          if(canonicalMention.label.isDefined && canonicalMention.label() != "O") matched = true
          canonical = canonicalMention.canonical()
        }
        
        if(matched){
      //    println("Canonical:" + canonical)

          for (men : PairwiseMention <-  mentionsOfEnt){
            val mention = mentionSeq(men.attr[ACEMentionIdentifiers].mId.replace("m", "").toInt)
            if(doc(mention.tokenEnd()).attr[NerTag].value == "O" || doc(mention.tokenEnd()).attr[NerTag].value.compareTo(doc( canonicalMention.tokenEnd()).attr[NerTag].value ) == 0 ) {    //todo:fix, pronoun, common noun

                mention.canonical := canonical
                if(canonicalMention.label.isDefined)
                  mention.label := canonicalMention.label()
            }

          }
        }
      }

    }

    predMap

  }

  def loadClassifier(model: String, classifierType: String): Object = {
    classifierType match {
      case Options.MaxEntClassifier | Options.PerceptronClassifier | Options.MarginPerceptronClassifier | Options.LbjPerceptronClassifier =>
        UiucCorefTrain.loadClassifier(classifierType, model)
      case _ => throw new Exception("unknown classifier type: " + classifierType)
    }
  }
}