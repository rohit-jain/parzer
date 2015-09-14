package edu.berkeley.nlp.io;

import java.io.File;
import java.io.FileFilter;
import java.util.List;
import java.util.ArrayList;

/**
 * Utilities for getting files recursively, with a filter.
 * 
 * @author Dan Klein
 */
public class IOUtils {
  public static List<File> getFilesUnder(String path, FileFilter fileFilter) {
    File root = new File(path);
    List<File> files = new ArrayList<File>();
    addFilesUnder(root, files, fileFilter);
    return files;
  }

  private static void addFilesUnder(File root, List<File> files, FileFilter fileFilter) {
    if (! fileFilter.accept(root)) return;
    if (root.isFile()) {
      files.add(root);
      return;
    }
    if (root.isDirectory()) {
      File[] children = root.listFiles();
      for (int i = 0; i < children.length; i++) {
        File child = children[i];
        addFilesUnder(child, files, fileFilter);
      }
    }
  }

}
