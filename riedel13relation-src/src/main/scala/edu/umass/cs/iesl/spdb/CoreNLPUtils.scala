package edu.umass.cs.iesl.spdb
import cc.factorie.app.nlp._
import java.util.ArrayList
import edu.stanford.nlp.ling._
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.util.CoreMap
import edu.stanford.nlp.trees.{Trees, PennTreebankLanguagePack, Tree => StanfordTree}
import scala.collection.JavaConversions._
import edu.stanford.nlp.ling.CoreAnnotations.{CharacterOffsetEndAnnotation, CharacterOffsetBeginAnnotation}
import cc.refectorie.user.jzheng.coref.PosTag
import java.lang.{Integer => Integer}
/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/11/12
 * Time: 12:16 PM
 * To change this template use File | Settings | File Templates.
 */

object CoreNLPUtils {
  /**
   * Converts document to an equivalent Stanford CoreNLP annotation object.
   */
  def toAnnotation(doc: cc.factorie.app.nlp.Document): Annotation = {
    require(doc.string != null)
    val result = new Annotation(doc.string)
    val tokens: java.util.List[CoreLabel] = new ArrayList[CoreLabel]

    //fill up id text
    result.set(classOf[CoreAnnotations.DocIDAnnotation],doc.name)
    result.set(classOf[CoreAnnotations.TextAnnotation],doc.string)

    //fill up tokens
    if (doc.tokens.size > 0) {
      for (token <- doc.tokens) {
        tokens.add(toCoreLabel(token))
      }
      result.set(classOf[CoreAnnotations.TokensAnnotation], tokens)
    }

    //fill up sentences
    if (doc.sentences.size > 0) {
      val sentences: java.util.List[CoreMap] = new ArrayList[CoreMap]
      for (sentence <- doc.sentences if sentence.length > 0) {
        sentences.add(toAnnotation(sentence, tokens))
      }
      result.set(classOf[CoreAnnotations.SentencesAnnotation], sentences)
    }
    result
  }

  /**
   * Converts a token to a corelabel
   */
  def toCoreLabel(token: Token): CoreLabel = {
    val label = new CoreLabel
    label.setBeginPosition(token.stringStart)
    label.setEndPosition(token.stringEnd)
    label.setIndex(token.position)
    label.setWord(token.string)
    if(token.attr[PosTag] != null)
      label.setTag(token.attr[PosTag].value)
    if(token.attr[NerTag] != null)
      label.setNER(token.attr[NerTag].value)
    if(token.attr[Lemma] != null)
      label.setLemma(token.attr[Lemma].value)
    label
  }

  /**
   * Converts a sentence into a Stanford CoreLabel annotation
   */
  def toAnnotation(sentence: cc.factorie.app.nlp.Sentence, docTokens: java.util.List[CoreLabel]): Annotation = {
    val result = new Annotation(sentence.string)
    result.set(classOf[CharacterOffsetBeginAnnotation], new Integer(sentence.start))
    result.set(classOf[CharacterOffsetEndAnnotation], new Integer(sentence.end))
    result.set(classOf[CoreAnnotations.TokenBeginAnnotation], new Integer(sentence.tokens.head.position))
    result.set(classOf[CoreAnnotations.TokenEndAnnotation], new Integer(sentence.tokens.last.position + 1))

    val sentenceTokens = docTokens.subList(
      sentence.tokens.head.position,
      sentence.tokens.last.position + 1)
    result.set(classOf[CoreAnnotations.TokensAnnotation], sentenceTokens)

    result
  }

  /**
   * Get the tokens of a document/sentence.
   */
  def getTokens(document: CoreMap): java.util.List[CoreLabel] = {
    document.get[java.util.List[CoreLabel], CoreAnnotations.TokensAnnotation](classOf[CoreAnnotations.TokensAnnotation])
  }

  /**
   * Get the sentences of a document.
   */
  def getSentences(document: CoreMap): java.util.List[CoreMap] = {
    document.get[java.util.List[CoreMap], CoreAnnotations.SentencesAnnotation](classOf[CoreAnnotations.SentencesAnnotation])
  }

  def getTree(sentence: CoreMap): StanfordTree = {
    sentence.get[StanfordTree, CoreAnnotations.TreeAnnotation](classOf[CoreAnnotations.TreeAnnotation])
  }
}