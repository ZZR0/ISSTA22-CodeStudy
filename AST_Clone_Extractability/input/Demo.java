class Demo {
  public void prepareReadBuffer(Configuration conf, Path file) throws IOException {
    // --- pre(i) ---
    FileSystem fs = file.getFileSystem(conf);
    long defaultLength = conf.getLong("dfs.blocksize", BLK);
    long bytesToProcess = 0;
    // --- body(i): Clone Instance ---
    FileStatus status = fs.getFileStatus(file);
    long fileLength = status.getLen();
    bytesToProcess = Math.max(fileLength, defaultLength);
    LOG.info(file.getName() + ": " + bytesToProcess);

    // --- post(i) ---
    byte[] buffer = new byte[(int) bytesToProcess];
    readData(fs, file, buffer);
  }
}
