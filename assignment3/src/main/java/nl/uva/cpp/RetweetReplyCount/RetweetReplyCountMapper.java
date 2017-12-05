package nl.uva.cpp.RetweetReplyCount;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

public class RetweetReplyCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	private final static IntWritable one = new IntWritable(1);
	private Text tag = new Text();

	static enum Counters {
		INPUT_RETWEETS,
		INPUT_REPLIES
		
	};

	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String string = value.toString().toLowerCase();
		String[] lines = string.split("\n");
		String tweet = lines[2];
		StringTokenizer itr = new StringTokenizer(tweet);
		int count = 0;

		// Flags to only count replies once in a tweet
		boolean reply = false;

		while (itr.hasMoreTokens()) {
			// Obtain the next token
			String token = itr.nextToken();
			
			// When the token is a retweet symbol
			if (isRetweet(token)) {
				// Increase the retweet counter by one
				context.getCounter(Counters.INPUT_RETWEETS).increment(1);
				if (itr.hasMoreTokens()) {
					token = itr.nextToken();
					// Check if the retweet actually points to a users tweet
					if (isReply(token)) {
						// Remove the @ from the username
						String user = token.substring(1);
						// The retweet is the rest of the tweet 
						String user_tweet = "";
						while (itr.hasMoreTokens()){
							user_tweet += " " + itr.nextToken();
						}
						tag.set(user  + user_tweet);
						context.write(tag, one);
					}
				}
			}
			if (isReply(token) && !reply) {
				context.getCounter(Counters.INPUT_REPLIES).increment(1);
				reply = true;
			}			
		}
	}

	private boolean isRetweet(String token) {
		// If the first two letters of a token are RT, then it's a retweet
		if (token.length() > 1 && token.substring(0,2).equals("rt")) {
			return true;
		}
		return false;
	}

	private boolean isReply(String token) {
		// It is a reply if the first letter of the token is a @
		if (token.substring(0,1).equals("@")) {
			return true;
		}
		return false;
	}
} 

