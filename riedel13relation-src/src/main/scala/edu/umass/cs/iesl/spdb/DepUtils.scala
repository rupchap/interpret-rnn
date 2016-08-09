package edu.umass.cs.iesl.spdb

import java.util.Stack
import collection.mutable.ArrayBuffer
import cc.factorie.app.nlp.parse._
import cc.factorie.app.nlp._
import cc.factorie.db.mongo.MongoCubbieCollection
import cc.refectorie.user.jzheng.coref.PosTag

/**
 * Created by IntelliJ IDEA.
 * User: lmyao
 * Date: 4/23/12
 * Time: 3:02 PM
 * To change this template use File | Settings | File Templates.
 */

object DepUtils {
  val FEASIBLEPOS = Set("NN", "NNP", "NNS", "NNPS", "VB", "VBD", "VBZ", "VBG", "VBN", "VBP", "RB", "RBR", "RBS", "IN", "TO", "JJ", "JJR", "JJS", "PRP", "PRP$")
  val JUNKLABELS = Set("conj", "ccomp", "parataxis", "xcomp", "pcomp", "advcl", "punct", "infmod", "cop", "abbrev", "neg", "det", "prt")     //I remove 'rcmod', Limin Yao , path staring with adv should be excluded
  val CONTENTPOS = Set("NN", "NNP", "NNS", "NNPS", "VB", "VBD", "VBZ", "VBG", "VBN", "VBP", "PRP", "PRP$", "RB", "RBR", "RBS", "JJ", "JJR", "JJS")
  val JUNKROOT = Set("say", "describe", "tell")
  val SUFFIX = Set("Jr.", "II", "Jr", "d.l.", "D.L.", "d.l", "D.L")

  // find all the ancestors of a node in order to find common ancestor of two nodes, called by find_path
  def find_ancestors(ancestors: Stack[Int], id: Int, depTree: ParseTree) {

    //due to parsing error, there are may be more than one roots, todo:
    //val rootEdge = depTree.modifiers(sentence.root).first

    var cur = id
    ancestors.push(cur)
    while (depTree.parentIndex(cur) != -1) {
      //sentence.root's index is -1
      cur = depTree.parentIndex(cur)
      ancestors.push(cur)
    }
  }


  //find the path from a source node to a dest node and the relationship, denoted by ->-><-<-  source != dest
  //return the rootOfPath and PathString
  def find_path(source: Int, dest: Int, depTree: ParseTree): (String, String, Seq[(Token, String)]) = {
    val tokens = depTree.sentence.tokens
    val srctmp = new Stack[Int] //source queue
    val srcStack = new Stack[Int]
    val destStack = new Stack[Int]
    val path = new ArrayBuffer[(Token, String)]
    find_ancestors(destStack, dest, depTree);
    find_ancestors(srcStack, source, depTree);

    var junk = false // for indicating junk path
    var res = ""
    //find common ancestor
    var common_ancestor: Int = -1
    while (!srcStack.empty && !destStack.empty && srcStack.peek == destStack.peek) {
      common_ancestor = srcStack.peek
      srcStack.pop
      destStack.pop
    }

    if (common_ancestor == -1) {
      path += ((depTree.rootChild, "exception"))
      return ("null", "exception", path)
    }

    //reverse the order of nodes in srcStack by a tmp stack
    while (!srcStack.empty) {
      val top = srcStack.pop
      srctmp.push(top)
    }

    //get nodes from srctmp, destStack and fill in path, todo:without source and dest node, find lemmas for each word using stanford pipeline annotator
    while (!srctmp.empty) {
      val top = srctmp.pop
      if (!FEASIBLEPOS.contains(tokens(top).attr[PosTag].value)) junk = true
      val label = depTree.label(top).categoryValue.toLowerCase
      if (JUNKLABELS.contains(label)) junk = true
      path += ((tokens(top), label))
      if (top != source) {
        if (tokens(top).attr[NerTag].value != "O") res += tokens(top).attr[NerTag].value + "<-" + label + "<-"
        else if(SUFFIX.contains(tokens(top).string)) res += tokens(top).attr[NerTag].value + "<-" + label + "<-"
        else  res += tokens(top).attr[Lemma].value + "<-" + label + "<-"
      }
      else
        res += "<-" + label + "<-"
    }
    path += ((tokens(common_ancestor), "rootOfPath")) //this is common ancestor of source and dest , // tokens(top).lemma, todo
    if(JUNKROOT.contains(tokens(common_ancestor).attr[Lemma].value.toLowerCase))  {junk = true;     }
    if (common_ancestor != source && common_ancestor != dest) {
      if (tokens(common_ancestor).attr[NerTag].value != "O") res += tokens(common_ancestor).attr[NerTag].value
      else if(SUFFIX.contains(tokens(common_ancestor).string)) res += tokens(common_ancestor).attr[NerTag].value
      else res += tokens(common_ancestor).attr[Lemma].value
    }
    while (!destStack.empty) {
      val top = destStack.pop
      if (!FEASIBLEPOS.contains(tokens(top).attr[PosTag].value)) {junk = true;  }
      val label = depTree.label(top).categoryValue.toLowerCase
      if (JUNKLABELS.contains(label)) {junk = true;    }
      path += ((tokens(top), label))
      if (top != dest) {
        //if (tokens(top).attr[NerTag].value != "O") junk = true //named entity should not appear on the path
        //if (SUFFIX.contains(tokens(top).string)) junk = true //suffix should not appear on the path
        if (tokens(top).attr[NerTag].value != "O") res += "->" + label + "->" + tokens(top).attr[NerTag].value
        else if(SUFFIX.contains(tokens(top).string)) res += "->" + label + "->" + tokens(top).attr[NerTag].value
        else res += "->" + label + "->" + tokens(top).attr[Lemma].value

      }
      else
        res += "->" + label + "->"
    }


    var content = false //path includes source and dest tokens, there should be content words between them
    val numContentWords = path.map(_._1).map(_.attr[PosTag].value).filter(CONTENTPOS.contains(_)).size

    if (numContentWords > 1) content = true // else {println("Junk content!")}
    if (!junk && !content) junk = true //heuristics for freebase candidate generation
    
    if(path.map(_._1).map(_.string).filter(_.matches("[\\?\"\\[]")).size > 0) junk = true

    if(path(0)._2 == "adv") junk = true
    if(path(0)._2 == "name" && path.reverse(0)._2 == "name") junk = true
    if(path(0)._2 == "sbj" && path.last._2 == "sbj") junk = true

    //filter out junk according to DIRT, rule 2, any dependency relation must connect two content words
    if (junk) {
      path += ((depTree.rootChild, "junk"))
      return ("null", "junk", path)
    } //comment this out for generating freebase instances

    (tokens(common_ancestor).attr[Lemma].value.toLowerCase, res.toLowerCase, path)
  }
  
  //find out mention of the SF entity in the sentence, output a mention
  def findOneArg(entstr : String, slot: Mention, sentence : Sentence) : Mention = {
    val words = entstr.split("[-\\s]")
    val abbr = words.map(_.substring(0,1)).mkString("")
    val candidates = new ArrayBuffer[Mention]
    val depTree = sentence.attr[ParseTree]
    for(token <- sentence.tokens.filter(_.attr[NerTag].value != "O")){
      if(entstr.contains(token.string) || (abbr.matches("^[A-Z]+$") && token.string.compareTo(abbr) == 0 ) ){  //second condition for handling abbr.
        var ancestor = token.sentencePosition
        while(depTree.parentIndex (ancestor) != -1  &&
          entstr.contains(sentence.tokens(depTree.parentIndex(ancestor)).string)
        )
        {
          ancestor = depTree.parentIndex(ancestor)
        }
        var tokenEnd = sentence.tokens(ancestor).position
        var tokenStart = token.position
        if(tokenEnd < tokenStart){
          val tmp = tokenEnd
          tokenEnd = tokenStart
          tokenStart = tmp
        }
        val mention = new Mention
        mention.docId := sentence.document.name
        mention.tokenBegin := tokenStart
        mention.tokenEnd := tokenEnd
        mention.charBegin := sentence.document.tokens(tokenStart).stringStart
        mention.charEnd := sentence.document.tokens(tokenEnd).stringEnd

        //mention.phrase := sentence.document.string.slice(mention.charBegin.value, mention.charEnd())
        mention.phrase := entstr
        candidates += mention
      }
    }
    
    val slotHead = EntityMentionUtils.headOfMention(slot,sentence)
    var maxDis = sentence.tokens.size
    var res : Mention = null
    for(mention <- candidates){
      val dist = math.abs(EntityMentionUtils.headOfMention(mention,sentence)-slotHead)
      if(dist < maxDis){
        maxDis = dist
        res = mention
      }
    }
    
//    //for debugging:
//    if(res != null)
//      println(entstr + "\t" + slot.phrase() + "\t" + res.phrase() )
    
//    if(res == null){
//      for(token <- sentence.tokens.filter(x => (x.attr[PosTag].value == "PRP$" || x.attr[PosTag].value == "PRP")) ) {//still some errors here, it can only be linked to non-person
//        val mention = new Mention
//        mention.charBegin := token.stringStart
//        mention.charEnd := token.stringEnd
//        mention.tokenBegin := token.position
//        mention.tokenEnd := token.position
//        //mention.phrase := token.string
//        mention.phrase := entstr
//        candidates += mention
//      }
//    }
//    maxDis = sentence.tokens.size
//    for(mention <- candidates){
//      val dist = math.abs(EntityMentionUtils.headOfMention(mention,sentence)-slotHead)
//      if(dist < maxDis){
//        maxDis = dist
//        res = mention
//      }
//    }
    return res
  }

  def findOneArg(entstr : String, sentence : Sentence) : Seq[Mention] = {
    val words = entstr.split("[-\\s]")
    val abbr = words.map(_.substring(0,1)).mkString("")
    val candidates = new ArrayBuffer[Mention]
    val depTree = sentence.attr[ParseTree]
    for(token <- sentence.tokens.filter(_.attr[NerTag].value != "O")){
      if(entstr.contains(token.string) || (abbr.matches("^[A-Z]+$") && token.string.compareTo(abbr) == 0 ) ){  //second condition for handling abbr.
        var ancestor = token.sentencePosition
        while(depTree.parentIndex (ancestor) != -1  &&
          entstr.contains(sentence.tokens(depTree.parentIndex(ancestor)).string)
        )
        {
          ancestor = depTree.parentIndex(ancestor)
        }
        var tokenEnd = sentence.tokens(ancestor).position
        var tokenStart = token.position
        if(tokenEnd < tokenStart){
          val tmp = tokenEnd
          tokenEnd = tokenStart
          tokenStart = tmp
        }
        val mention = new Mention
        mention.docId := sentence.document.name
        mention.tokenBegin := tokenStart
        mention.tokenEnd := tokenEnd
        mention.charBegin := sentence.document.tokens(tokenStart).stringStart
        mention.charEnd := sentence.document.tokens(tokenEnd).stringEnd

        mention.phrase := sentence.document.string.slice(mention.charBegin(), mention.charEnd())
        mention.canonical := entstr
        candidates += mention
      }
    }

    return candidates
  }
}