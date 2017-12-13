package nl.uva.cpp.ReplyCount;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

public class ReplyCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	private final static IntWritable one = new IntWritable(1);
	private Text user = new Text();

	static enum Counters {
		REPLIERS
	};

	// This one maps the most replied user
	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String string = value.toString().toLowerCase();
		String[] lines = string.split("\n");
		String tweet = lines[2];
		// Remove the twitter URL from the name
		String replier = lines[1].replace("u	http://twitter.com/","");
		StringTokenizer itr = new StringTokenizer(tweet);
		int count = 0;

		// Flags to only count replies once in a tweet
		boolean reply = false;

		while (itr.hasMoreTokens()) {
			// Obtain the next token
			String token = itr.nextToken();
			// the tweet is a reply, we count one reply per tweet
			if (isReply(token) && !reply) {
				// Increase the reply counter by one
				user.set(replier);
				context.write(user, one);
				context.getCounter(Counters.REPLIERS).increment(1);
			}
		}
	}

	private boolean isReply(String token) {
		// It is a reply if the first letter of the token is a @
		if (token.substring(0,1).equals("@")) {
			return true;
		}
		return false;
	}
} 

