package nl.uva.cpp.ReplyCount;

import org.apache.hadoop.conf.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.*;

public class ReplyCountTool extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = this.getConf();
		conf.set("textinputformat.record.delimiter", "\n\n");

		Job job = Job.getInstance(conf);
		job.setJarByClass(this.getClass());

		// Set the input and output paths for the job, to the paths given
		// on the command line.
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		// Use our mapper and reducer classes.
		job.setMapperClass(ReplyCountMapper.class);
		job.setReducerClass(ReplyCountReducer.class);

		// Our input file is a text file.
		job.setInputFormatClass(TextInputFormat.class);

		// Our output is a mapping of text to integers. (See the tutorial for
		// some notes about how you could map from text to text instead.)
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		// Limit the number of reduce/map classes to what was specified on
		// the command line.
		int numTasks = Integer.valueOf(args[2]);
		job.setNumReduceTasks(numTasks);
		job.getConfiguration().setInt("mapred.max.split.size", 750000 / numTasks);
		job.waitForCompletion(true);
		// This limits the number of running mappers, but not the total.
		// job.getConfiguration().setInt("mapreduce.job.running.map.limit", numTasks);
		Job job2 = Job.getInstance(conf);
		job2.setJarByClass(this.getClass());

		// Set the input and output paths for the job, to the paths given
		// on the command line.
		FileInputFormat.setInputDirRecursive(job2, true);

		// 
		FileInputFormat.addInputPath(job2, new Path(args[0]));

		// Use our mapper and reducer classes.
		job2.setMapperClass(ReplierCountMapper.class);
		job2.setReducerClass(ReplyCountReducer.class);

		// Our input file is a text file.
		job2.setInputFormatClass(TextInputFormat.class);

		// Our output is a mapping of text to integers. (See the tutorial for
		// some notes about how you could map from text to text instead.)
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);

		// Limit the number of reduce/map classes to what was specified on
		// the command line.
		job2.setNumReduceTasks(numTasks);
		job2.getConfiguration().setInt("mapred.max.split.size", 750000 / numTasks);
		// Run the job!
		return job2.waitForCompletion(true) ? 0 : 1;
	}
}
