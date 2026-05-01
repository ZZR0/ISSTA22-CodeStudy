# Generating Positive & Negative Samples for Clone Detection

## **High-level overview**

| Item | BigCloneBench (`./data/train.txt`) | NiCad Camel (`nicad_camel_clone_data.jsonl`, Sim: 0.7) |
|---|---:|---:|
| Total functions | 7,302 | 2,739 |
| Clone groups | N/A  | 985 groups |
| Total possible pairs (Universe) | N/A | 3,749,691 |
| Positive pairs (Label = 1) | 450,862 | 4,068 (max possible) |
| Negative pairs (Label = 0) | 450,166 | 3,745,623 (max possible) |

---

This guide explains how we prepare training data for our code clone detection models.  
Specifically, it details how we calculate and generate **Positive Samples** (clones) and **Negative Samples** (non-clones) from our dataset.

---

## 1. Math Background: Counting Pairs

To create a training dataset, we need to form "pairs" of functions. Understanding how many unique pairs exist in a group is crucial for balancing our data.

---

### The "Handshake" Analogy (Combinatorics)

Imagine a Clone Group containing **4 functions** (A, B, C, D). Since they are all clones of each other, every function must be paired with every *other* function. This forms a **Complete Graph**.

---

### How to calculate the number of pairs

1. **Connections:** Each of the 4 functions can connect to 3 other functions (itself excluded).

$$
4 \times 3 = 12 \text{ connections}
$$

2. **Removing Duplicates:** A connection from A to B is the same as B to A. We must divide by 2 to remove this double counting.

$$
\frac{12}{2} = 6 \text{ unique pairs}
$$

---

### The General Formula

For any group of size $N$, the maximum number of unique pairs is calculated using the "N choose 2" formula:

$$
\text{Total Pairs} = \binom{N}{2} = \frac{N(N-1)}{2}
$$

---

### Example

- **Clone Group Size:** 5 functions  
- **Calculation:**

$$
\frac{5 \times 4}{2} = 10 \text{ pairs}
$$

---

## 2. Types of Samples

### ✅ Positive Samples (Label 1)

-   **Definition:** Pairs of functions that are **semantic clones** (do the same thing).
-   **Source:** All possible pairs **within** the same Clone Group.
-   **Calculation:** We sum the pairs formula for every clone group in the dataset.

### ❌ Negative Samples (Label 0)

-   **Definition:** Pairs of functions that are **functionally different**.
-   **Source:** Pairs formed by picking functions from **different** Clone Groups.
-   **Calculation:** The total capacity for negative samples is huge (Total Universe of Pairs - Positive Pairs).
-   **Balancing:** Since negative pairs vastly outnumber positive ones, we **randomly sample** a specific number of negative pairs to match the number of positive pairs (1:1 ratio) to prevent training bias.

---

## 3. How to Run the Script

We use the `gen_neg_clone_sample.py` script to automatically calculate these statistics and generate the balanced negative samples.

### Prerequisites

Ensure you have run `gen_init_train_sample.py` first to generate unique `func_id`s for the dataset.

### Command

```bash
python gen_neg_clone_sample.py

```

### Script Output (Console)

The script will print a detailed breakdown of the math for the current dataset:

```text
(base_py311) √ DataProc % py gen_neg_clone_sample.py  
--- Loading data from: data/nicad_camel_clone_data.jsonl ---
Loaded 985 groups with 2739 total functions.

=== Detailed Calculation of Dataset Capacity ===
1. Total Functions (N): 2739
   - We have 2739 unique functions in the dataset.
   - Total possible pairs (Universe) = N * (N - 1) / 2
   - 2739 * 2738 / 2 = 3,749,691 pairs

2. Max Possible Positive Samples (Clones): 4,068
   - Calculation: Sum of pairs within each clone group.
   - Formula: Sum( size * (size - 1) / 2 ) for all groups.
   - This represents all pairs where Label = 1.

3. Max Possible Negative Samples (Non-Clones): 3,745,623
   - Calculation: Universe - Positives
   - Formula: 3,749,691 - 4,068 = 3,745,623
   - This represents all pairs where Label = 0.
================================================

--- Generating 4068 Negative Samples (Balanced) ---
Done. Generated 4068 unique negative pairs.
JSONL saved to: data/nicad_camel_neg_samples.jsonl
TXT saved to: data/nicad_camel_neg_samples.txt (4068 lines)
HTML report generated: display_neg_sample.html
Markdown report generated: data/display_neg_sample.md

--- Verifying Negative Samples: data/nicad_camel_neg_samples.txt ---
Total Lines Checked: 4068
Valid Negative Pairs: 4068
✅ SUCCESS: All pairs are valid negative samples (different clone groups).
```

### Generated Files

1. **`data/nicad_camel_neg_samples.txt`**: The final training list. Format:

```text
(base_py311) √ DataProc % head -n 5 data/nicad_camel_neg_samples.txt
1624_1400	119_107	0
1497_1337	1108_943	0
1195_1022	1033_884	0
188_190	2437_1911	0
2637_2273	1739_1451	0

```

2. **`data/display_neg_sample.md`**: A Markdown table for quick inspection on GitHub.
3. **`display_neg_sample.html`**: A full HTML page with side-by-side code comparison.

---


## 4. Visual Inspection (Generated Report)

The script generates a Markdown report to help you verify the quality of the negative samples. You can click the link below to view the actual generated pairs:

*(The file above `data/display_neg_sample.md` is automatically updated every time you run the script)*

### 🔍 Negative Sample Inspection (generated)

| Pair ID | Function A (Source) | Function B (Target) |
| :--- | :--- | :--- |
| **#1** | <b>ID:</b> `11_6`<br>`org.apache.camel.component.xmlsecurity.api.DefaultXmlSignature2Message.omitXmlDeclaration(Message message, Input input)`<br><pre>    protected Boolean omitXmlDeclaration(Message message, Input input) {<br>        Boolean omitXmlDeclaration = message.getHeader(XmlSignatureConstants.HEADER_OMIT_XML_DECLARATION, Boolean.class);<br>        if (omitXmlDeclaration == null) {<br>            omitXmlDeclaration = input.omitXmlDeclaration();<br>        }...</pre> | <b>ID:</b> `9_1`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheEndpoint.doStart()`<br><pre>    protected void doStart() throws Exception {<br>        super.doStart();<br>        cache = CamelContextHelper.lookup(getCamelContext(), cacheName, Cache.class);<br>        if (cache == null) {<br>            if (configuration.isCreateCacheIfNotExist()) {...</pre> |
| **#2** | <b>ID:</b> `13_11`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.setResult(Message message, boolean success, Object result, Object oldValue)`<br><pre>    private void setResult(Message message, boolean success, Object result, Object oldValue) {<br>        message.setHeader(CaffeineConstants.ACTION_SUCCEEDED, success);<br>        message.setHeader(CaffeineConstants.ACTION_HAS_RESULT, oldValue != null &#124;&#124; result != null);<br><br>        if (oldValue != null) {...</pre> | <b>ID:</b> `12_8`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.getValue(final Message message, final String type)`<br><pre>    private Object getValue(final Message message, final String type) throws Exception {<br>        Object value = message.getHeader(CaffeineConstants.VALUE);<br>        if (value == null) {<br>            if (type != null) {<br>                Class<?> clazz = getEndpoint().getCamelContext().getClassResolver().resolveClass(type);...</pre> |
| **#3** | <b>ID:</b> `13_11`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.setResult(Message message, boolean success, Object result, Object oldValue)`<br><pre>    private void setResult(Message message, boolean success, Object result, Object oldValue) {<br>        message.setHeader(CaffeineConstants.ACTION_SUCCEEDED, success);<br>        message.setHeader(CaffeineConstants.ACTION_HAS_RESULT, oldValue != null &#124;&#124; result != null);<br><br>        if (oldValue != null) {...</pre> | <b>ID:</b> `11_5`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.getKey(final Message message)`<br><pre>    private Object getKey(final Message message) throws Exception {<br>        String value;<br>        value = message.getHeader(CaffeineConstants.KEY, String.class);<br>        if (value == null) {<br>            value = configuration.getKey();...</pre> |
| **#4** | <b>ID:</b> `13_10`<br>`org.apache.camel.component.ehcache.EhcacheProducer.setResult(Message message, boolean success, Object result, Object oldValue)`<br><pre>    private void setResult(Message message, boolean success, Object result, Object oldValue) {<br>        message.setHeader(EhcacheConstants.ACTION_SUCCEEDED, success);<br>        message.setHeader(EhcacheConstants.ACTION_HAS_RESULT, oldValue != null &#124;&#124; result != null);<br><br>        if (oldValue != null) {...</pre> | <b>ID:</b> `11_6`<br>`org.apache.camel.component.xmlsecurity.api.DefaultXmlSignature2Message.omitXmlDeclaration(Message message, Input input)`<br><pre>    protected Boolean omitXmlDeclaration(Message message, Input input) {<br>        Boolean omitXmlDeclaration = message.getHeader(XmlSignatureConstants.HEADER_OMIT_XML_DECLARATION, Boolean.class);<br>        if (omitXmlDeclaration == null) {<br>            omitXmlDeclaration = input.omitXmlDeclaration();<br>        }...</pre> |
| **#5** | <b>ID:</b> `11_4`<br>`org.apache.camel.component.caffeine.load.CaffeineLoadCacheProducer.getKey(final Message message)`<br><pre>    private Object getKey(final Message message) throws Exception {<br>        String value = message.getHeader(CaffeineConstants.KEY, String.class);<br>        if (value == null) {<br>            value = configuration.getKey();<br>        }...</pre> | <b>ID:</b> `13_11`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.setResult(Message message, boolean success, Object result, Object oldValue)`<br><pre>    private void setResult(Message message, boolean success, Object result, Object oldValue) {<br>        message.setHeader(CaffeineConstants.ACTION_SUCCEEDED, success);<br>        message.setHeader(CaffeineConstants.ACTION_HAS_RESULT, oldValue != null &#124;&#124; result != null);<br><br>        if (oldValue != null) {...</pre> |
| **#6** | <b>ID:</b> `10_2`<br>`org.apache.camel.component.caffeine.load.CaffeineLoadCacheProducer.onInvalidateAll(Message message)`<br><pre>    public void onInvalidateAll(Message message) {<br>        Set<?> keys = message.getHeader(CaffeineConstants.KEYS, Set.class);<br>         <br>        if (keys == null) {<br>            cache.invalidateAll();...</pre> | <b>ID:</b> `13_10`<br>`org.apache.camel.component.ehcache.EhcacheProducer.setResult(Message message, boolean success, Object result, Object oldValue)`<br><pre>    private void setResult(Message message, boolean success, Object result, Object oldValue) {<br>        message.setHeader(EhcacheConstants.ACTION_SUCCEEDED, success);<br>        message.setHeader(EhcacheConstants.ACTION_HAS_RESULT, oldValue != null &#124;&#124; result != null);<br><br>        if (oldValue != null) {...</pre> |
| **#7** | <b>ID:</b> `11_5`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.getKey(final Message message)`<br><pre>    private Object getKey(final Message message) throws Exception {<br>        String value;<br>        value = message.getHeader(CaffeineConstants.KEY, String.class);<br>        if (value == null) {<br>            value = configuration.getKey();...</pre> | <b>ID:</b> `13_10`<br>`org.apache.camel.component.ehcache.EhcacheProducer.setResult(Message message, boolean success, Object result, Object oldValue)`<br><pre>    private void setResult(Message message, boolean success, Object result, Object oldValue) {<br>        message.setHeader(EhcacheConstants.ACTION_SUCCEEDED, success);<br>        message.setHeader(EhcacheConstants.ACTION_HAS_RESULT, oldValue != null &#124;&#124; result != null);<br><br>        if (oldValue != null) {...</pre> |
| **#8** | <b>ID:</b> `9_1`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheEndpoint.doStart()`<br><pre>    protected void doStart() throws Exception {<br>        super.doStart();<br>        cache = CamelContextHelper.lookup(getCamelContext(), cacheName, Cache.class);<br>        if (cache == null) {<br>            if (configuration.isCreateCacheIfNotExist()) {...</pre> | <b>ID:</b> `10_2`<br>`org.apache.camel.component.caffeine.load.CaffeineLoadCacheProducer.onInvalidateAll(Message message)`<br><pre>    public void onInvalidateAll(Message message) {<br>        Set<?> keys = message.getHeader(CaffeineConstants.KEYS, Set.class);<br>         <br>        if (keys == null) {<br>            cache.invalidateAll();...</pre> |
| **#9** | <b>ID:</b> `12_8`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.getValue(final Message message, final String type)`<br><pre>    private Object getValue(final Message message, final String type) throws Exception {<br>        Object value = message.getHeader(CaffeineConstants.VALUE);<br>        if (value == null) {<br>            if (type != null) {<br>                Class<?> clazz = getEndpoint().getCamelContext().getClassResolver().resolveClass(type);...</pre> | <b>ID:</b> `11_5`<br>`org.apache.camel.component.caffeine.cache.CaffeineCacheProducer.getKey(final Message message)`<br><pre>    private Object getKey(final Message message) throws Exception {<br>        String value;<br>        value = message.getHeader(CaffeineConstants.KEY, String.class);<br>        if (value == null) {<br>            value = configuration.getKey();...</pre> |


**Note:** The HTML report (`display_neg_sample.html`) ([link](https://github.com/unoselab/ISSTA22-CodeStudy/blob/master/DataProc/display_neg_sample.html)) provides a more detailed side-by-side view with syntax highlighting support if opened in a browser.
