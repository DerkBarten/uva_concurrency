package nl.uva.cpp.RetweetCount;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.omg.PortableInterceptor.USER_EXCEPTION;

public class RetweetCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	private final static IntWritable one = new IntWritable(1);
	private Text tag = new Text();

	static enum Counters {
		RETWEETERS
	};

	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String string = value.toString().toLowerCase();
		String[] lines = string.split("\n");
		String tweet = lines[2];
		// Remove the twitter URL from the name
		String retweeter = lines[1].replace("u	http://twitter.com/","");
		StringTokenizer itr = new StringTokenizer(tweet);
		int count = 0;

		// Flags to only count replies once in a tweet
		boolean reply = false;

		while (itr.hasMoreTokens()) {
			// Obtain the next token
			String token = itr.nextToken();
			if (isRetweet(token)) {
				// Increase the retweet counter by one
				if (itr.hasMoreTokens()) {
					token = itr.nextToken();
					// Check if the retweet actually points to a users tweet
					if (isReply(token)) {
						// Remove the @ from the username
						String user = token.substring(1);
						tag.set(user);
						context.write(tag, one);
						context.getCounter(Counters.RETWEETERS).increment(1);
					}
				}
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
		if (token.substring(0,1).equals("@") && token.length() > 1) {
			return true;
		}
		return false;
	}
} 

