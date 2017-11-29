package nl.uva.cpp.ReplyCount;

import org.apache.hadoop.conf.*;
import org.apache.hadoop.util.*;

public class ReplyCount {

	public static void main(String[] args) {
		int retval = 0;
		try {
			// first job should count the users that receive the most replies
			retval = ToolRunner.run(new Configuration(), new ReplyCountTool(), args);
		} catch (Exception e) {
			System.out.println(e.getMessage());
			retval = 2;			
		}
		System.exit(retval);	
	}
}