package edu.umass.cs.iesl.spdb

import cc.factorie.app.nlp.Document
import collection.mutable.ArrayBuffer
import cc.factorie.app.nlp.parse.ParseTree
import cc.factorie.app.strings.Stopwords

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/27/12
 * Time: 4:06 PM
 * To change this template use File | Settings | File Templates.
 */

object Features {
  def extractFeatures(source : Mention, dest : Mention, doc : Document, srcner : String,  dstner : String) : ArrayBuffer[String] = {
    val features = new ArrayBuffer[String]
    val sentence = doc.tokens(source.tokenBegin()).sentence

    val shead = EntityMentionUtils.headOfMention(source,sentence)
    val dhead = EntityMentionUtils.headOfMention(dest,sentence)
    val (rootOfPath, pathString, path) = DepUtils.find_path(shead, dhead, sentence.attr[ParseTree])
    if (pathString != "junk" && pathString != "exception" && path.size < 10) {
      val pathContentWords = path.drop(1).dropRight(1).filter(_._1.attr[NerTag].value == "O").map(_._1.attr[Lemma].value.toLowerCase).filter(!Stopwords.contains(_)) // only none-ner tokens can be trigger words

    //  val trigger = if (pathContentWords.size == 0) rootOfPath else pathContentWords.mkString(",")

     // features += "trigger#" + trigger
      //features += "source#" +  EntityMentionUtils.wordPOS(source,sentence)
      //features += "dest#" + EntityMentionUtils.wordPOS(dest,sentence)
      if(pathString.length()>0)
        features += "path#" + pathString

    } else if (math.abs(dhead - shead) > 10){
      return features
    }

    features += "type#" + srcner + "->" + dstner

    //between context, newly added
    if(dest.tokenBegin() - source.tokenEnd() < 11) {
      val lexContext = EntityMentionUtils.betweenLexContext(source,dest, sentence)
      val posContext = EntityMentionUtils.betweenPOSContext(source,dest, sentence)
      if(lexContext.filter(_.length()>0).size > 0){
        features ++= lexContext.filter(_.length()>0).map(x => "lex#"+x)
        features ++= posContext.filter(_.length()>0).map(x => "pos#"+x)

        features += "lex-ner#" + srcner  + "|" + lexContext(0) + "|" + dstner
        features += "pos-ner#" + srcner  + "|" + posContext(0) + "|" + dstner
      }
    }

    //lc, rc, words to the left of the source arg, to the right of the dest arg
   /* val neighborContext = EntityMentionUtils.neighbor(source,dest, sentence)
    if(neighborContext.length() > 0) {
      val contexts = neighborContext.split("\t")
      features ++= contexts
    } */

    features.filter(_.length()>0).filter(x => !(x.contains("javascript")))
  }
}