### Exemplary Java Function (25 Tokens)

```java
public int findMax(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

```

### Token Breakdown

In natural language processing for code, tokens include keywords, identifiers, operators, and separators. Here is how a tokenizer sees the function above:

1. `public`
2. `int`
3. `findMax`
4. `(`

...

22. `}`
23. `else`
24. `{`
25. `return` ... (Note: Adding the final `b`, `;`, and `}` would bring the count slightly higher, but the logic above represents the typical "short block" used in clone detection datasets).

### Why this is good for your dataset:

- **Logical Density:** Even with very few tokens, it contains a conditional branch (`if-else`), which is critical for testing if a model understands logic rather than just text.
- **Variable Mapping:** It uses two parameters, allowing you to test if the model can distinguish between `(a, b)` and `(b, a)` clones.
- **Standard Structure:** It follows the standard boilerplate that LLMs expect for Java methods.

---

### Exemplary Java Function (30 Tokens)

```java
public double calculateCircleArea(double radius) {
    if (radius <= 0) {
        return 0.0;
    }
    double pi = 3.14159;
    return pi * radius * radius;
}
```

---

### Exemplary Java Function (40 - 50 Tokens)

```java
public double calculateAverage(int[] numbers) {
    if (numbers == null || numbers.length == 0) {
        return 0.0;
    }
    double sum = 0;
    for (int num : numbers) {
        sum += num;
    }
    return sum / numbers.length;
}
```

---

### Exemplary Java Function (50 - 60 Tokens)

```java
public int linearSearch(int[] array, int target) {
    if (array == null || array.length == 0) {
        return -1;
    }
    for (int i = 0; i < array.length; i++) {
        if (array[i] == target) {
            return i;
        }
    }
    return -1;
}
```

---

### Exemplary Java Function (50 - 60 Tokens)

```java
public boolean push(int[] stack, int element, int top) {
    if (top >= stack.length - 1) {
        System.out.println("Stack Overflow");
        return false;
    }
    stack[top + 1] = element;
    return true;
}
```

---

### Exemplary Java Function (70 - 80 Tokens)

```java
public String reverseAndUpper(String input) {
    if (input == null || input.isEmpty()) {
        return "";
    }
    StringBuilder sb = new StringBuilder();
    for (int i = input.length() - 1; i >= 0; i--) {
        char c = input.charAt(i);
        sb.append(Character.toUpperCase(c));
    }
    return sb.toString();
}
```

---

### Exemplary Java Function (100 Tokens)

```java
public Map<String, Integer> countWordFrequency(String[] words) {
    if (words == null || words.length == 0) {
        return new HashMap<>();
    }
    Map<String, Integer> frequencyMap = new HashMap<>();
    for (String word : words) {
        if (word == null) continue;
        String lowerWord = word.toLowerCase();
        int count = frequencyMap.getOrDefault(lowerWord, 0);
        frequencyMap.put(lowerWord, count + 1);
    }
    return frequencyMap;
}

```

### Token Breakdown

Using a standard BPE tokenizer (like **CodeGPT** or **Tiktoken**), the segments are categorized as follows:

1. `public`
2. `Map`
3. `<`
4. `String`
5. `,`
6. `Integer`

..

100. `;`
101. `}`
102. `return`
103. `frequencyMap`
104. `;`
105. `}`

_(Note: While the raw word count is lower, the inclusion of Generics `<String, Integer>` and repeated map calls significantly increases the **token density**. Most models will perceive this as an 80–95 token block.)_

### Why this is effective for your dataset:

- **Generic Handling:** This tests if the model understands the `<T, K>` syntax.
- **Library Methods:** It uses `getOrDefault`, a specific Java 8+ API. In clone detection, identifying that this is semantically equivalent to an `if (map.containsKey())` check is a classic **Type-4 (Semantic)** challenge.
- **Flow Control:** The `continue` keyword is introduced here, testing if the model understands non-linear loop progression.

---

### Exemplary Java Function (~90 Tokens)

```java
public String getFileExtension(String filePath, boolean includeDot) {
    if (filePath == null || filePath.lastIndexOf('.') == -1) {
        return "";
    }
    int lastDotIndex = filePath.lastIndexOf('.');
    int lastSeparatorIndex = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
    if (lastSeparatorIndex > lastDotIndex) {
        return "";
    }
    String extension = filePath.substring(includeDot ? lastDotIndex : lastDotIndex + 1);
    return extension.toLowerCase();
}

```

### Exact Token Count Breakdown

Using the logic of a standard code tokenizer (where punctuation and operators are distinct tokens), here is the exact count:

| Category         | Tokens                                                                                                                                                                      | Count  |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **Header**       | `public`, `String`, `getFileExtension`, `(`, `String`, `filePath`, `,`, `boolean`, `includeDot`, `)`                                                                        | 10     |
| **Block Start**  | `{`                                                                                                                                                                         | 1      |
| **Safety Check** | `if`, `(`, `filePath`, `==`, `null`, `                                                                                                                                      |        |
| **Return Empty** | `{`, `return`, `""`, `;`, `}`                                                                                                                                               | 5      |
| **Dot Indexing** | `int`, `lastDotIndex`, `=`, `filePath`, `.`, `lastIndexOf`, `(`, `'.'`, `)`, `;`                                                                                            | 10     |
| **Sep Indexing** | `int`, `lastSeparatorIndex`, `=`, `Math`, `.`, `max`, `(`, `filePath`, `.`, `lastIndexOf`, `(`, `'/'`, `)`, `,`, `filePath`, `.`, `lastIndexOf`, `(`, `'\\'`, `)`, `)`, `;` | 22     |
| **Logic Check**  | `if`, `(`, `lastSeparatorIndex`, `>`, `lastDotIndex`, `)`, `{`, `return`, `""`, `;`, `}`                                                                                    | 11     |
| **Substring**    | `String`, `extension`, `=`, `filePath`, `.`, `substring`, `(`, `includeDot`, `?`, `lastDotIndex`, `:`, `lastDotIndex`, `+`, `1`, `)`, `;`                                   | 16     |
| **Final Return** | `return`, `extension`, `.`, `toLowerCase`, `(`, `)`, `;`                                                                                                                    | 7      |
| **Block End**    | `}`                                                                                                                                                                         | 1      |
| **Total**        |                                                                                                                                                                             | **98** |

_(Note: In raw "word" counts, this is much lower, but for LLM training, every symbol—`.`, `(`, `?`, `:`, `+`—is a discrete token. Depending on whether `lastIndexOf` is split into `last`, `Index`, `Of`, the count typically settles between **90 and 105**.)_

### Research Utility for Clone Detection

This specific example is excellent for your dataset because:

1. **Ternary Operator:** It uses `? :`, allowing you to test if the model can recognize this as semantically identical to an `if-else` block.
2. **Overloaded Logic:** It uses `Math.max` and multiple `lastIndexOf` calls, which challenges the model's ability to track variable values across library calls.
3. **Cross-Platform Handling:** It addresses both `/` and `\` separators, a common pattern in secure software engineering.

---

To maximize the performance of your LLM-based clone detector, the construction of **Negative Samples (Label 0)** must go beyond simple random selection. The goal is to provide "Hard Negatives" that challenge the model to look past surface-level similarities.

### Strategies for Constructing Negative Samples (Label 0)

* **1. Random Shuffling (Easy Negatives)**
* **Method:** Pair functions randomly from entirely different projects.
* **Purpose:** Provides a baseline for the model to distinguish between different domains (e.g., a UI function vs. a math utility).


* **2. Intra-Project/File Pairing (Hard Negatives)**
* **Method:** Select two different functions from the **same Java file** or the same package.
* **Why it works:** Functions in the same file often share the same author, variable naming conventions, and imported libraries.
* **Impact:** Forces the model to ignore coding style and focus purely on the **logic and control flow**.


* **3. Near-Miss Selection (Threshold-Based)**
* **Method:** Use non-LLM tools (Deckard/iClone) to find pairs with **low-to-moderate similarity scores** (e.g., a Deckard similarity of 0.5) that did not meet the threshold for a clone.
* **Impact:** These pairs often have similar AST structures but perform different tasks, improving the model's **precision**.



---

### Recommended Data Distribution

To build a robust dataset, I recommend the following composition for your Negative Samples:

| Category | Proportion | Source | Learning Objective |
| --- | --- | --- | --- |
| **Easy Negatives** | 30% | Cross-project random pairs | Basic domain separation. |
| **Hard Negatives** | 70% | **Same file/package** pairs | Deep semantic discrimination. |

---

### Conceptual Workflow

1. **Positive (1):** Pairs confirmed as clones by iClone/Deckard.
2. **Negative (0):** Pairs from the same directory that were **not** flagged by the tools.

By focusing on **Intra-project Hard Negatives**, you ensure that your LLM doesn't "cheat" by simply looking at the vocabulary, but instead learns to analyze the actual program structure.
