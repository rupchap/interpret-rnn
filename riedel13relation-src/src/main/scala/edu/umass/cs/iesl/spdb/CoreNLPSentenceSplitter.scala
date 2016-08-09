package edu.umass.cs.iesl.spdb
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling._
import edu.stanford.nlp.util.CoreMap
import cc.factorie.app.nlp.{Document, Sentence, Token  }
import java.util.Properties
import scala.collection.JavaConversions._
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation
import java.lang.{Integer => Integer}
import java.util.regex.Pattern


/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/11/12
 * Time: 2:35 PM
 * tokenization and sentence split
 * \n is definitely a sentence boundary, according to doc.text. After tokenization, this information may be missing. that's why Limin modifies the code
 *
 */

object CoreNLPTokenizer  {
  val tokenizer = new CoreNLPPTBTokenizer

  def createTokens(doc: Document, tokens: java.util.List[CoreLabel], posBias : Int) {
    for (token <- tokens)
      new Token(doc, token.beginPosition+posBias, token.endPosition+posBias)
  }
  
  def createSentences(doc : Document,  sentences: java.util.List[CoreMap], posBias : Int){
    for (sentence <- sentences) {
      for (token: CoreLabel <- sentence.get[java.util.List[CoreLabel], TokensAnnotation](classOf[TokensAnnotation])) {
        new Token(doc, token.beginPosition+posBias, token.endPosition + posBias)
      }
      val begin = sentence.get[Integer, CoreAnnotations.TokenBeginAnnotation](
        classOf[CoreAnnotations.TokenBeginAnnotation])
      val end = sentence.get[Integer, CoreAnnotations.TokenEndAnnotation](
        classOf[CoreAnnotations.TokenEndAnnotation])
      val senLen = end - begin
      new Sentence(doc,doc.tokens.size - senLen,senLen)
    }
  }

  /*//todo:headline should be included  <HEADLINE>
  Halliburton CEO will oversee new Dubai headquarters
    </HEADLINE>          */
  def annotate(doc: Document) = {
    var text = doc.string
    if(text.contains("<HEADLINE>")) extractHeadLine(text,0,doc)
    if(text.contains("<P>")) extractP(text,0,doc)
    else if(text.contains("<POST>")) extractPost(text,0,doc)
    else if(text.contains("<TURN>")) extractTurn(text,0,doc)
  }
  
  def extractHeadLine(text : String, start : Int, doc : Document){
    var startPos = text.indexOf("<HEADLINE>")
    var splitPos = text.indexOf("</HEADLINE>")
    if(startPos != -1) startPos += 10
    val stringSen = text.substring(startPos,splitPos)
    if( stringSen.length > 0 && stringSen.compareTo("\n")!=0) {
      val sentences = tokenizer.doOneSentence(stringSen)
      createSentences(doc,sentences, startPos)
    }
  }

  //+4 to skip </P>, +3 to skip <P>
  def extractP(text : String,  start : Int, doc : Document){
    var startPos = text.indexOf("<P>", start)
    var splitPos = text.indexOf("</P>", startPos)
    var stringSen = ""
    while( startPos != -1 && splitPos != -1 ){
      startPos += 3
      stringSen = text.substring(startPos,splitPos)
      if(stringSen.contains("\n\n")){
        extractLine(text,startPos,doc)
      }
      else if( stringSen.length > 0 && stringSen.compareTo("\n")!=0) {
        val sentences = tokenizer.doOneSentence(stringSen)
        createSentences(doc,sentences, startPos)
      }
      startPos = text.indexOf("<P>", splitPos + 4)
      splitPos = text.indexOf("</P>", startPos)
    }
  }

  //+11 to skip </POSTDATE>, +7 to skip </POST>
  def extractPost(text : String,  start : Int,  doc : Document){
    var startPos = text.indexOf("</POSTDATE>", start)
    var splitPos = text.indexOf("</POST>", startPos)
    var stringSen = ""

    while(startPos != -1 && splitPos != -1 && stringSen.length() < 50000){
      startPos += "</POSTDATE>".length()
      stringSen = text.substring(startPos,splitPos)
      if(stringSen.contains("\n\n")){
        extractLine(stringSen,startPos, doc)
      } 
      else if( stringSen.length > 0 && stringSen.compareTo("\n")!=0) {
        val sentences = tokenizer.doOneSentence(stringSen)
        createSentences(doc,sentences, startPos)
      }
      startPos = text.indexOf("</POSTDATE>", splitPos+"</POST>".length())
      splitPos = text.indexOf("</POST>", startPos)
    }
  }

  //+11 to skip </POSTDATE>, +5 to skip </POST>
  def extractTurn(text : String,  start : Int,  doc : Document){
    var startPos = text.indexOf("</SPEAKER>", start)
    var splitPos = text.indexOf("</TURN>", startPos)
    var stringSen = ""

    while(startPos != -1 && splitPos != -1 && stringSen.length() < 50000){
      startPos += "</SPEAKER>".length()
      stringSen = text.substring(startPos,splitPos)
      if(stringSen.contains("\n\n")){
        extractLine(stringSen,startPos, doc)
      }
      else if( stringSen.length > 0 && stringSen.compareTo("\n")!=0) {
        val sentences = tokenizer.doOneSentence(stringSen)
        createSentences(doc,sentences, startPos)
      }
      startPos = text.indexOf("</SPEAKER>", splitPos+"</TURN>".length())
      splitPos = text.indexOf("</TURN>", startPos)
    }
  }
  
  def extractLine(text : String,  bias: Int,  doc: Document){
    var startPos = 0
    var splitPos = text.indexOf("\n\n", startPos)
    var stringSen = ""
    while(splitPos != -1){
      stringSen = text.substring(startPos,splitPos)
      if( stringSen.length > 0 && stringSen.compareTo("\n")!=0) {
        val sentences = tokenizer.doOneSentence(stringSen)
        createSentences(doc,sentences, startPos+bias)
      }
      startPos = splitPos + 2
      splitPos = text.indexOf("\n\n", startPos)
    }

    //this is the last sentence
    stringSen = text.substring(startPos,text.length())
    if( stringSen.length > 0 && stringSen.compareTo("\n")!=0) {
      val sentences = tokenizer.doOneSentence(stringSen)
      createSentences(doc,sentences, startPos+bias)
    }
  }
}

//usage:
//val tokenizer = new CoreNLPPTBTokenizer
//tokenizer.annotate(document)
class CoreNLPPTBTokenizer {
  val tokenizer = new  PTBTokenizerAnnotator()
  val props = new Properties()
  props.put("annotators", "tokenize, ssplit")
  val pipeline = new StanfordCoreNLP(props)

  def annotate(doc : Document){
    val sentences = doOneSentence(doc.string)
    for (sentence <- sentences; if(sentence.size()>0)) {
        for (token: CoreLabel <- sentence.get[java.util.List[CoreLabel], TokensAnnotation](classOf[TokensAnnotation])) {
          new Token(doc, token.beginPosition, token.endPosition)
        }
        val begin = sentence.get[Integer, CoreAnnotations.TokenBeginAnnotation](
          classOf[CoreAnnotations.TokenBeginAnnotation])
        val end = sentence.get[Integer, CoreAnnotations.TokenEndAnnotation](
          classOf[CoreAnnotations.TokenEndAnnotation])
        new Sentence(doc,begin, end-begin)

    }
  }

  def doOneSentence(text:String):java.util.List[CoreMap] = {
    if(text.contains("<POSTDATE>")) {
      println(text)
    }
    val a = new Annotation(text)
    pipeline.annotate(a)
    val sentences = CoreNLPUtils.getSentences(a)
    sentences
  }
}