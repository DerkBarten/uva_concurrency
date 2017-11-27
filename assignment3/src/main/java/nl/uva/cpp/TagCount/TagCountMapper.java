package nl.uva.cpp.TagCount;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

public class TagCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	private final static IntWritable one = new IntWritable(1);
	private Text tag = new Text();

	static enum Counters {
		INPUT_TAGS
	}

	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String string = value.toString().toLowerCase();
		String[] lines = string.split("\n");
		String tweet = lines[2];
		StringTokenizer itr = new StringTokenizer(tweet);

		int count = 0;
		while (itr.hasMoreTokens()) {
			// Obtain the next token
			String token = itr.nextToken();
			
			// Check if word is hashtag
			if (isHastag(token)) {
				tag.set(token);
				// Write (tag, 1) as (key, value) in output
				context.write(tag, one);
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
} 

