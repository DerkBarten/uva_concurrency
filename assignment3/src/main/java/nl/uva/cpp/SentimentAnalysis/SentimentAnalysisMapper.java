package nl.uva.cpp.SentimentAnalysis;

import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Properties;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import me.champeau.ld.UberLanguageDetector;
import  edu.stanford.nlp.ling.CoreAnnotations;
import  edu.stanford.nlp.pipeline.Annotation;
import  edu.stanford.nlp.pipeline.StanfordCoreNLP;
import  edu.stanford.nlp.neural.rnn.RNNCoreAnnotations ;;
import  edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import  edu.stanford.nlp.trees.Tree;
import  edu.stanford.nlp.util.CoreMap;

public class SentimentAnalysisMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	private final static IntWritable one = new IntWritable(1);
	private IntWritable sentiment;
	private Text tag = new Text();
	String parseModelPath = "englishPCFG.ser.gz";
	String sentimentModelPath = "sentiments.ser.gz";
	UberLanguageDetector detector = UberLanguageDetector.getInstance();


	static enum Counters {
		INPUT_TAGS
	}
	

	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String string = value.toString().toLowerCase();
		String[] lines = string.split("\n");
		String tweet = lines[2];
		StringTokenizer itr = new StringTokenizer(tweet);
		String language;

		boolean hasHashtag = false;

		int count = 0;
		while (itr.hasMoreTokens()) {
			// Obtain the next token
			String token = itr.nextToken();
			
			// Check if word is hashtag
			if (isHastag(token)) {
				if (!hasHashtag) {
					language = detector.detectLang(tweet);
					if (!language.equals("en")){
						break;
					}
					sentiment = new IntWritable(findSentiment(tweet)); 
					hasHashtag = true;
				}
				System.out.println(token);
				tag.set(token);
				// Write (tag, 1) as (key, value) in output
				context.write(tag, sentiment);
				// Increment a counter.
				context.getCounter(Counters.INPUT_TAGS).increment(1);
				
			}
		}
	}

	private boolean isHastag(String token) {
		// 1. should have hashtag at start of string
		// 2. should be longer at least 2 chars
		if (token.substring(0,1).equals("#") && token.length() > 1) {
			return true;
		} 
		// 3. if string is length 2, second char can't be a number
		return false;
	}

	private  int  findSentiment(String  text) {
		Properties  props = new  Properties ();
		props.setProperty("annotators", "tokenize , ssplit , parse , sentiment");
		props.put("parse.model", parseModelPath);
		props.put("sentiment.model", sentimentModelPath);

		StanfordCoreNLP  pipeline = new  StanfordCoreNLP(props);
		int  mainSentiment = 0;
		if (text != null && text.length () > 0) {
			int  longest = 0;
			Annotation  annotation = pipeline.process(text);
			for (CoreMap  sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
				Tree  tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class );
				int  sentiment = RNNCoreAnnotations.getPredictedClass(tree);
				String  partText = sentence.toString ();
				if (partText.length () > longest) {
					mainSentiment = sentiment;
					longest = partText.length ();
				}
			}
		}
		//This  method  is very  demanding  so try so save  some  memory
		pipeline = null;
		System.gc();
		return  mainSentiment;
	}

} 

