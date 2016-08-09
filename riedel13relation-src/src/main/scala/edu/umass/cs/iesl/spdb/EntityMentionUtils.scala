package edu.umass.cs.iesl.spdb

import cc.factorie.app.nlp._
import cc.factorie.db.mongo.MongoCubbieCollection
import parse.ParseTree
import collection.mutable.ArrayBuffer
import cc.refectorie.user.jzheng.coref.PosTag

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/25/12
 * Time: 3:10 PM
 * To change this template use File | Settings | File Templates.
 */

object EntityMentionUtils {

  val NERLABELS = Set("PERSON", "ORGANIZATION", "LOCATION", "MISC", "NUMBER", "DATE")

  def betweenLexContext(m1:Mention, m2:Mention, sentence : Sentence) : ArrayBuffer[String] = {
    val start = m1.tokenEnd() + 1
    val end = m2.tokenBegin()
    val res = new ArrayBuffer[String]
    //   if(end - start < 10)
    
    res += sentence.document.tokens.slice(start,end).filter(_.attr[NerTag].value=="O").map(_.string).mkString(" ")
    res += sentence.document.tokens.slice(start,end).map(_.string).mkString(" ")

    //   else ""
    res
  }

  def betweenPOSContext(m1:Mention, m2:Mention, sentence : Sentence) : ArrayBuffer[String] = {
    val res = new ArrayBuffer[String]
    val start = m1.tokenEnd() + 1
    val end = m2.tokenBegin()
    //   if(end - start < 10)
    //res += sentence.document.tokens.slice(start,end).filter(_.attr[NerTag].value=="O").map(_.attr[PosTag].value).mkString(" ")
    res += sentence.document.tokens.slice(start,end).map(_.attr[PosTag].value).mkString(" ")
    //   else ""
    res
  }

  def neighbor(m1:Mention, m2:Mention, sentence : Sentence) : String = {
    
    var lcontext = ""

    var index = sentence.document.tokens(m1.tokenBegin()).sentencePosition
    if(index - 2 > -1)
      lcontext += "lc#" + sentence.tokens.slice(index -2, index).filter(_.attr[NerTag].value=="O").map(_.string).mkString(" ")
    else if(index -1 > -1 && sentence.tokens(index-1).attr[NerTag].value == "O")
      lcontext += "lc#" + sentence.tokens(index-1).string

    var rcontext = ""
    index = sentence.document.tokens(m2.tokenEnd()).sentencePosition
    if(index + 2 < sentence.tokens.length)
      rcontext += "rc#" + sentence.tokens.slice(index+1,index+3).filter(_.attr[NerTag].value=="O").map(_.string).mkString(" ")
    else if(index + 1 < sentence.tokens.length && sentence.tokens(index+1).attr[NerTag].value=="O")
      rcontext += "rc#" + sentence.tokens(index+1).string

    if(lcontext.length > 0 && rcontext.length() > 0) lcontext + "\t" + rcontext
    else if(lcontext.length>0) lcontext
    else if(rcontext.length()>0) rcontext
    else ""
  }
 
  //"PRP", "PRP$",
  //require that ner is done , no entity linking
  def findAllSimpleMentions(sentence : Sentence, mentions:MongoCubbieCollection[Mention]) = {
    //need to test whether the last mention is added
    var lastNER = "O"
    var lastStart = 0
    
    val tokens = sentence.tokens
    for (index <- 0 until tokens.length) {
      val token = tokens(index)
      val ner = token.attr[NerTag].value
      if (ner == "O") {
        //a mention ended
        if (lastNER != "O" && NERLABELS.contains(lastNER)) {
          val mention = new Mention
          mention.docId := sentence.document.name
          mention.charBegin := tokens(lastStart).stringStart
          mention.charEnd := tokens(index-1).stringEnd
          mention.tokenBegin := tokens(lastStart).position
          mention.tokenEnd := tokens(index-1).position
          mention.phrase := sentence.document.string.substring(mention.charBegin(),mention.charEnd()) //sentence.doc.tokens.slice (mention.tokenBegin(),mention.tokenEnd()+1).map(_.string).mkString(" ")
          if(mention.phrase().contains("\n")) mention.canonical := mention.phrase().replaceAll ("\n", " ") else mention.canonical := mention.phrase()
          mention.label := lastNER
          mentions += mention
        }
      } else {
        if (lastNER != ner) {
          if (lastNER != "O" && NERLABELS.contains(lastNER)) {
            val mention = new Mention
            mention.docId := sentence.document.name
            mention.charBegin := tokens(lastStart).stringStart
            mention.charEnd := tokens(index-1).stringEnd
            mention.tokenBegin := tokens(lastStart).position
            mention.tokenEnd := tokens(index-1).position
            mention.phrase := sentence.document.string.substring (mention.charBegin(),mention.charEnd())
            if(mention.phrase().contains("\n")) mention.canonical := mention.phrase().replaceAll ("\n", " ") else mention.canonical := mention.phrase()
            mention.label := lastNER
            mentions += mention
//            var entity : Entity = null
//            if (entities.query(_.canonical(mention.phrase())).hasNext) entity = entities.query(_.canonical(mention.phrase())).next()
//            else {
//              entity = new Entity
//              entity.id = new ObjectId
//              entity.canonical := mention.phrase()
//              entities += entity
//            }
//            mention.entity ::= entity
          }
          lastStart = index
        }
      }
      lastNER = ner
    }

    //add common noun and pronoun mentions
   /* val document = sentence.document
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
        mentions += mention
      }

      if (token.attr[NerTag].value.compare("O") == 0 && token.attr[PosTag].value.startsWith("N")) {
        val mention = new Mention
        mention.docId := document.name
        mention.tokenBegin := token.position
        mention.tokenEnd := token.position
        mention.charBegin := token.stringStart
        mention.charEnd := token.stringEnd
        mention.phrase := document.string.substring(mention.charBegin(),mention.charEnd())
        mention.canonical := "None"
        mentions += mention
      }
    } */
  }


  def headOfMention(mention: Mention, sentence : Sentence): Int = {
    // find out the head of this mention
    var head = 0
    val depTree = sentence.attr[ParseTree]
    val untilTokenPosition = sentence.document.tokens(mention.tokenEnd()).sentencePosition + 1
    val beginTokenPosition = sentence.document.tokens(mention.tokenBegin()).sentencePosition
    if (untilTokenPosition - beginTokenPosition == 1) head = beginTokenPosition
    else {
      //find each token's ancestors which are in the entitymention, choose the rightmost one
      head = beginTokenPosition

      for (i <- beginTokenPosition until untilTokenPosition) {
        var ancestor = i
        while(depTree.parentIndex(ancestor) < untilTokenPosition && depTree.parentIndex(ancestor) >= beginTokenPosition)
          ancestor = depTree.parentIndex(ancestor)
        if (ancestor > head) head = ancestor
      }
    }
    head
  }
 
  //in order to access text of a mention, we need to load the document first
  def wordPOS(mention: Mention, sentence : Sentence) = {
    sentence.document.slice(mention.tokenBegin(),mention.tokenEnd()+1).map(t => t.string + "/" + t.attr[PosTag].value + "/" + t.attr[NerTag].value).mkString(" ")
  }
}