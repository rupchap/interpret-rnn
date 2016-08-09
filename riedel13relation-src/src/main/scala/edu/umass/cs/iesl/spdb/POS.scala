package edu.umass.cs.iesl.spdb
import cc.factorie.app.nlp._
import java.io.FileInputStream
import scala.collection.JavaConversions._
import edu.stanford.nlp.pipeline.{MorphaAnnotator, NERCombinerAnnotator, POSTaggerAnnotator}
import cc.refectorie.user.jzheng.coref.PosTag

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/11/12
 * Time: 12:01 PM
 * To change this template use File | Settings | File Templates.
 */

class CoreNLPTagger(modelFile: String)  {
  val tagger = new POSTaggerAnnotator(modelFile, true)

  def annotate(doc: Document) = {
    val annotation = CoreNLPUtils.toAnnotation(doc)

    tagger.annotate(annotation)

    for (token <- CoreNLPUtils.getTokens(annotation)) {
      doc.tokens(token.index).attr += new PosTag(token.tag)
    }

  }
}

object POS extends CoreNLPTagger("edu/stanford/nlp/models/pos-tagger/wsj3t0-18-left3words/left3words-distsim-wsj-0-18.tagger")

class NerTag(val value : String)

class CoreNLPNERTagger(modelFile: String){
  val tagger = new NERCombinerAnnotator(false, modelFile)

  def annotate(doc: Document) = {
    val annotation = CoreNLPUtils.toAnnotation(doc)

    tagger.annotate(annotation)

    for (token <- CoreNLPUtils.getTokens(annotation)){
      doc.tokens(token.index).attr += new NerTag(token.ner)
    }

  }
}

object NERTagger extends CoreNLPNERTagger("edu/stanford/nlp/models/ner/conll.distsim.crf.ser.gz")

class Lemma(val value : String)

class CoreNLPMorphTagger {
  val tagger = new MorphaAnnotator(false)

  def annotate(doc: Document) = {
    val annotation = CoreNLPUtils.toAnnotation(doc)

    tagger.annotate(annotation)

    for (token <- CoreNLPUtils.getTokens(annotation)) {
      doc.tokens(token.index).attr += new Lemma(token.lemma)
    }

  }
}

trait TokenPOSCubbie extends TokenCubbie {
  val postag = StringSlot("postag")

  override def storeToken(t:Token): this.type = {
    super.storeToken(t)
    postag := t.attr[PosTag].value
    this
  }
  override def fetchToken: Token = {
    val t = super.fetchToken
    t.attr += new PosTag(postag.value)
    t
  }
}

trait TokenNerCubbie extends TokenCubbie {
  val ner = StringSlot("ner")

  override def storeToken(t:Token): this.type = {
    super.storeToken(t)
    ner := t.attr[NerTag].value
    this
  }
  override def fetchToken: Token = {
    val t = super.fetchToken
    t.attr += new NerTag(ner.value)
    t
  }
}

trait TokenLemmaCubbie extends TokenCubbie {
  val lemma = StringSlot("lemma")

  override def storeToken(t:Token): this.type = {
    super.storeToken(t)
    lemma := t.attr[Lemma].value
    this
  }
  override def fetchToken: Token = {
    val t = super.fetchToken
    t.attr += new Lemma(lemma.value)
    t
  }
}