package nl.uva.cpp.ReplyCount;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

public class ReplierCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	private final static IntWritable one = new IntWritable(1);
	private Text user = new Text();

	static enum Counters {
		REPLIERS
	};

	// This one maps the most user that replied the most
	@Override
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		String string = value.toString().toLowerCase();
		String[] lines = string.split("\n");
		String tweet = lines[2];
		
		// we want to find the combinations of repliers and repliants

		// from the previous job we know the most replied to
		// now we want to find which user replies to most to this specific user
	}

	private boolean isReply(String token) {
		// It is a reply if the first letter of the token is a @
		if (token.substring(0,1).equals("@")) {
			return true;
		}
		return false;
	}
} 