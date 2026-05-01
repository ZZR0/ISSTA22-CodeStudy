import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class LogFilter {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java LogFilter <logfile>");
            return;
        }

        String filePath = args[0];
        // 정규표현식: 숫자 + %| 패턴 찾기
        Pattern pctPattern = Pattern.compile("(\\d+)%\\|");
        boolean skippedFlag = false;

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                
                Matcher m = pctPattern.matcher(line);
                boolean isProgressBar = m.find() && (line.contains("it/s]") || line.contains("s/it]"));

                if (isProgressBar) {
                    // 매칭된 숫자(퍼센트) 파싱
                    int pct = Integer.parseInt(m.group(1));

                    // 0~10% 이거나 90~100% 면 출력
                    if (pct <= 10 || pct >= 90) {
                        System.out.println(line);
                        skippedFlag = false;
                    } else {
                        // 11~89% 구간은 생략 메시지 (한 번만)
                        if (!skippedFlag) {
                            System.out.println("   ... [ Middle Progress Skipped (11% ~ 89%) ] ...");
                            skippedFlag = true;
                        }
                    }
                } else {
                    // 일반 텍스트 라인은 무조건 출력
                    System.out.println(line);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
    }
}