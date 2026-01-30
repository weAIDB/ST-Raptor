table_example_en = r"""{
    "2023 City-Level Departmental Overall Budget Performance Target Table": {
        "Department Name": "Zhanjiang Municipal Human Resources and Social Security Bureau",
        "Basic Information": [
            {
                "Number of Personnel Funded by the Government": "750",
                "Number of Subordinate Secondary Units": "10"
            }
        ],
        "Overall Budget Situation": {
            "Department Budget Expenditure": "",
            "Budget Amount (Ten Thousand Yuan)": "",
            "Source of Income": ""
        },
        "Overall Performance Target": "",
        "Annual Key Work Tasks": {
            "Name": "",
            "Main Implementation Content": "",
            "Planned Investment (Ten Thousand Yuan)": "",
            "Expected Goals to be Achieved (Summary)": ""
        },
        "Other Tasks to Be Completed (Optional)": "",
        "Performance Indicators": {
            "First-Level Indicator": "",
            "Second-Level Indicator": "",
            "Third-Level Indicator": "",
            "Indicator Value": ""
        },
    }
}"""

query_example = """
Example Query 1: How much funding does the government plan to invest in urban and rural resident pension insurance?

Example Primitive Sentence 1: [CHL] + [Annual Key Tasks]

Explanation 1: You can find the relevant information under the "Annual Key Tasks" entry within the table, which will help you refine your search.

Example Query 2: What are the first-level indicators within the performance metrics?

Example Primitive Sentence 2: [EXT] + [Performance Indicators] + [First-Level Indicators]

Explanation 2: You can locate the "Performance Indicators" and "First-Level Indicators" cells in the table; their intersection will provide the answer.
"""

primitive_prompt_fewshot_en = """
### Instruction
You are given a table with a nested structure. The Table Content section contains the actual data for the table. A question related to this table is provided under ### Query.

Tools include the available primitive statements. You need to use these statements step by step to obtain relevant table data that can help answer the question. When generating the statements, you must consider both the table structure and the nature of the question.

The Table Content provides the actual data for the table. You can narrow down the range of the data needed to answer the question by generating appropriate statements.

Examples contain sample tables, questions, and the expected primitive statements you should generate, along with detailed explanations for why those statements are generated. Your generated statements should follow the format and logic in these examples.

Other Query Answers contains answers to other sub-questions.

### Tools
Child Node Extraction Statement: To extract all successor nodes of a specific node, output [CHL] + [KEY]
Example: [CHL] + [Name]

Parent Node Extraction Statement: To extract the parent node of a specific node, output [FAT] + [KEY]
Example: [FAT] + [Zhang San]

Data Extraction Statement: To extract the value at a specific position based on row and column, output [EXT] + [ROW_KEY] + [COLUMN_KEY]
Example: [EXT] + [A] + [B] retrieves the value at the intersection of row A and column B.

End Data Extraction Statement: If the data obtained so far can already answer the question, output [END]
Example: [END]

### Examples
{table_example}
{query_example}

### Query
{query}

### Table Content
{table}

### Other Query Answers
{query_history}

### Note
Note! Every part of the operations must be enclosed in []!
The goal is to obtain data that can help answer the question, not to directly answer the question. Do not include any additional explanation or extra quotation marks. Based on the current state, output the next tool to be used, and output only one step at a time!
If the Table Content already contains enough information to answer the question, output [END] to conclude!
"""

primtive_prompt_condition_en = """
### Instruction
You are given a table with a nested structure. The Table Content section provides the actual data in the table. A question related to this table is given under ### Query.

Your task is to identify the filtering conditions on the table content implied by the question. If the question does not require filtering the table, simply output None.

Refer to the Examples section to guide the format and logic of your response.

### You may use the following comparison operators:

== : equal to

>= : greater than or equal to

<= : less than or equal to

> : greater than

< : less than

!= : not equal to

in : substring containment

Your output format should follow: [COND] + [Column Name] + [Operator] + [Condition Value]

### Examples:

To find entries where the name is Zhang San: [COND] + [Name] + [==] + [Zhang San]

To filter items that contain the word “pencil” in the product column: [COND] + [Product] + [in] + [Pencil]

To find all deposit records in 2016:

[COND] + [Year] + [==] + [2016]  
[COND] + [Operation] + [==] + [Deposit]  

### Query
{query}

### Table Content
{table}

### Note
If no filtering is required to answer the question, output None.
If multiple filtering conditions are needed, output each on a separate line.
Do not include any additional explanation — only output the condition statements.
"""

primitive_prompt_math_en = """
### Instruction
You are given a table with a nested structure. Table Content provides the actual data in the table, and a question about it is given under ### Query.

Your task is to analyze the question and determine whether it requires any mathematical operation to be performed on the table's data. There are five types of operations you may use, listed below.

If no mathematical operation is required to answer the question, simply output None.

### Available operations:

CNT – Count

SUM – Summation

AVR – Average

MIN – Minimum

MAX – Maximum

Your output format should follow: [MATH] + [Column Name] + [Operation]

### Examples:

To count all male students: [MATH] + [Gender] + [CNT]

To count the number of entries in the information mapping table: [MATH] + [Entry] + [CNT]

To calculate the total output value of a company: [MATH] + [Output Value] + [SUM]

To calculate the total cost of produced goods: [MATH] + [Cost] + [SUM]

To get the average monthly output value: [MATH] + [Monthly Output Value] + [AVR]

To find the student with the minimum age: [MATH] + [Age] + [MIN]

To find the student with the maximum age: [MATH] + [Age] + [MAX]

To find the project with the highest cost: [MATH] + [Cost] + [MAX]

### Query
{query}

### Table Content
{table}

### Note
Important: If the question does not require any of the five supported mathematical operations, output None.
Do not include any additional explanation — only output the operation statement.
"""

primitive_prompt_zeroshot_en = """
### Instruction
You are given a table with a nested structure and a question about it, labeled as ### Query.
Tools lists the available primitive operations. Please use these primitives to retrieve the relevant data from the table step by step in order to answer the question.
Your use of primitives should be guided by both the structure of the table and the form of the question.

Table Content provides the actual data in table a. Use the primitives to progressively narrow down the scope needed to answer the question.

Examples provide sample inputs including a table, a question, the expected primitive operations, and detailed explanations for why each primitive is used. Your output should follow the style of these examples.

Other Query Answers contains answers to other sub-questions, which may be relevant for reasoning or reuse.

### Tools
Child Node Extraction Primitive: To extract all child nodes of a given node, output: [CHL] + [KEY]
Example: [CHL] + [Name]

Parent Node Extraction Primitive: To extract the parent node of a given key, output: [FAT] + [KEY]
Example: [FAT] + [Zhang San]

Data Extraction Primitive: To extract a value based on row and column keys, output: [EXT] + [ROW_KEY] + [COLUMN_KEY]
Example: [EXT] + [A] + [B] retrieves the value at the intersection of the row containing A and the column containing B.

End of Extraction Primitive: If the currently retrieved data is sufficient to answer the question, output: [END]
Example: [END]

### Query
{query}

### Table Content
{table}

### Other Query Answers
{query_history}

### Note
Important: Every part of the operation must be enclosed in square brackets []!
The goal is to acquire the data needed to answer the question — not to directly answer it.
Do not include any additional explanation, formatting, or quotation marks.
Based on the current state, output the next primitive to use — only one step at a time.
If the data in the table is already sufficient to answer the question, simply output [END].
"""

query_decompose_prompt_en = """
### Instruction
You are given a nested-structure table and a question about it. Please decompose the given question into multiple simpler sub-questions and output a list of questions.
Note: You only need to decompose questions that involve parallel operations — such as those containing conjunctions like "and," "or," "respectively," or simple arithmetic operations like sum or difference over a few items.
You should also determine whether each sub-question requires retrieval from the table, or whether it can be answered using the results of previous sub-questions.
If a sub-question requires table lookup, output True; otherwise, output False.

### Nested Schema
{schema}

### Query
{query}

### Example
Input: What is the sum of salaries in 2021 and 2022?
Output:
[Query] Retrieve the salary for 2021. [Retrieve] True
[Query] Retrieve the salary for 2022. [Retrieve] True
[Query] Calculate the sum of 2021 and 2022 salaries. [Retrieve] False

Input: What is the difference between salaries in 2021 and 2022?
Output:
[Query] Retrieve the salary for 2021. [Retrieve] True
[Query] Retrieve the salary for 2022. [Retrieve] True
[Query] Calculate the difference between 2021 and 2022 salaries. [Retrieve] False

Input: What is Zhang San’s age?
Output:
[Query] What is Zhang San’s age? [Retrieve] True

Input: What is the major with the fewest junior college graduates in the School of Sociology and Law?
Output:
[Query] What is the major with the fewest junior college graduates in the School of Sociology and Law? [Retrieve] True

Input: What category do utility fees under management costs and office equipment depreciation under depreciation expenses belong to, respectively?
Output:
[Query] What category do utility fees under management costs belong to? [Retrieve] True
[Query] What category does office equipment depreciation under depreciation expenses belong to? [Retrieve] True
[Query] What category do utility fees under management costs and office equipment depreciation under depreciation expenses belong to, respectively? [Retrieve] False

### Note
Do not include any additional explanation or annotations. Output one sub-question per line in the specified format.
If no decomposition is needed, output the original question in the specified format.
"""

entity_extract_prompt_en = """
### Instruction
You are given a table and a question about the table. Please extract the nouns from the question that are likely to be used for retrieving information from the table, and return them as a Python list.

### Table Schema
{schema}

### Query
{query}

### Note
Do not provide any additional explanation or annotations. Only output a Python list.
"""


back_verification_prompt_en = """
### Instruction
You are given a table in JSON format, a question based on the table, and a possible answer. Based on the table and the provided answer, generate {n} different questions that could reasonably lead to the given answer.

Please output exactly {n} questions, one per line, without any additional formatting.

### Table
{table}

### Query
{query}

### Answer
{answer}

### Note
Only output one question per line. Do not include numbering or any other formatting.
"""

check_answer_prompt_en = """
### Instruction
You are given a question and an answer generated by a large language model. If the answer conveys any of the following meanings — "insufficient data to answer," "no data available for reference," or "unable to answer this question" — output F. Otherwise, output T.

### Query
{query}

### Answer
{answer}

### Note
Output T or F only. Do not include any additional formatting.
"""

semantic_reasoning_prompt_en = """
### Instruction
You are given the following data. Please answer the question based on this data.

### Evidence
{evidence}

### Query
{query}

### Note
Do not provide any explanation or annotations. Output a single answer only.
"""


direct_table_reasoning_prompt_en = """
### Instruction
You are given a table represented in JSON format. Please answer the following question based on the table.

### Table
{table}

### Query
{query}

### Note
Do not provide any additional explanation or annotations. Output a single answer only.
"""

evaluation_prompt_en = """
### Instruction
You are given two values, A and B, and both are answers to the same question. A is the correct answer, and B is another person's answer. Determine whether B is correct based on A. Output T if B is correct, and F if it is incorrect.
If A and B are numerically equal but have different units (e.g., 498 vs. 498 billion yuan, 98 vs. 98 people), consider B correct.

### Input
A: {a}
B: {b}

### Note
Output T or F only. T indicates the answer is correct, and F indicates it is incorrect. Do not provide any explanation or additional formatting.
"""

primitive_prompt_old = """
### Instruction
现在有一个具有嵌套结构的表格，有一个针对该表格提出的问题，记作 ### Query。
### Tools是提供的原语句，请你使用这些原语句，一步一步的获取可供回答该问题的相关表格数据。你的原语句生成需要既考虑表格的结构，又考虑问题的形式。
{type_string}
### Examples是提供的样例，包含表格，问题，和期望你生成的元语句，并且配有为何生成这个元语句的详细的说明。你的元语句生成应该仿照这些样例的形式。
### Iteration History 中提供了当前子问题的历史查询信息。
### Other Query Answers 记录了其他子问题的答案。

### Tools
1. 子节点提取原语句：提取某结点的所有后继节点，请输出 [CHL] + [KEY]
    例：[CHL] + [姓名]
2. 父节点提取原语句：提取某结点的父节点，请输出 [FAT] + [KEY]
    例：[FAT] + [张三]
3. 数据提取原语：根据行列提取某个位置的值，请输出 [EXT] + [ROW_KEY] + [COLUMN_KEY]
    例：[EXT] + [A] + [B] 获取内容A所在行，内容B所在列交叉位置的值。
4. 条件选择原语句：使用条件获取某结点的后继节点，请输出 [COND] + [KEY] + [Python Function]
    判断代码需要使用def进行定义，具有一个参数，返回布尔值，代表是否选择该条目。
    例：[COND] + [姓名] + [def cond(x): return x == '张三']
5. 比较运算原语句：比较两个值的大小，请输出 [CMP] + [VALUE1] + [VALUE2] + [Python Function]
    比较代码需要使用def进行定义，具有两个参数，返回比较的结果。
    例：[CMP] + [123] + [124] + [def cmp(x, y): return x > y]
6. 逻辑控制原语：将某个函数应用在某结点的所有后继节点上，清输出 [FOREACH] + [KEY] + [Python Function]
    代码需要使用def进行定义，具有一个参数，返回操作结果。
    例：[FOREACH] + [年] + [def foreach(x): return x + 1]
7. 结束数据获取原语句：如果当前获得的数据已经可以回答问题，请输出 [END]
    例：[END]
        
### Examples

{EXP_TAB}

{EXP_QUERY}

### Query
{query}

{table_string}

### Other Query Answers
{query_history}

### Note
注意！输出的操作的每一部分都要使用[]括起来！
目的是获得可以回答问题的数据，而不是直接回答问题！你不需要输出额外的解释，不需要任何额外的标注引号，请你基于当前状态输出下一步要使用的工具，请你仅输出一步！
如果Table内容已经可以回答问题，请直接输出[END]结束！
"""


table_example = r"""Table Content:
{
"2023年市级部门整体预算绩效目标表": {
    "部门名称": "湛江市人力资源和社会保障局",
    "基本信息": [
        {
            "财政供养人员数": "750",
            "下属二级单位数": "10"
        }
    ],
    "预算整体情况": {
            "部门预算支出": "",
            "预算金额（万元）": "",
            "收入来源": ""
    },
    "总体绩效目标": "",
    "年度重点工作任务":{
            "名称": "",
            "主要实施内容": "",
            "拟投入的资金（万元）": "",
            "期望达到的目标 （概述）": ""
    },
    "其他需完成的任务 （可选填）": "",
    "绩效指标": {
            "一级指标": "",
            "二级指标": "",
            "三级指标": "",
            "指标值": ""
    },
}
}"""

query_example = """
Example Query 1: 政府拟在城乡居民养老保险上投入多少资金？
Example Primitive Sentence 1: [CHL] + [年度重点工作任务]
Explanation 1: 你可以在表格中年度重点工作任务的条目下找到相关的信息，从而进一步进行搜索。

Example Query 2: 绩效指标里的第一级指标有哪些？
Example Primitive Sentence 2: [EXT] + [绩效指标] + [一级指标]
Explanation 2: 你可以在表格里找到绩效指标和一级指标这两个单元格，他们交叉的地方就是答案。
"""

primitive_prompt_fewshot = """
### Instruction
现在有一个具有嵌套结构的表格，有一个针对该表格提出的问题，记作 ### Query。
Tools是提供的原语句，请你使用这些原语句，一步一步的获取可供回答该问题的相关表格数据。你的原语句生成需要既考虑表格的结构，又考虑问题的形式。
Table Content 给定了表格a的具体数据。你可以通过生成原语句缩小寻找问题答案的范围。
Examples是提供的样例，包含表格，问题，和期望你生成的元语句，并且配有为何生成这个元语句的详细的说明。你的元语句生成应该仿照这些样例的形式。
Other Query Answers 记录了其他子问题的答案。

### Tools
1. 子节点提取原语句：提取某结点的所有后继节点，请输出 [CHL] + [KEY]
    例：[CHL] + [姓名]
2. 父节点提取原语句：提取某结点的父节点，请输出 [FAT] + [KEY]
    例：[FAT] + [张三]
3. 数据提取原语：根据行列提取某个位置的值，请输出 [EXT] + [ROW_KEY] + [COLUMN_KEY]
    例：[EXT] + [A] + [B] 获取内容A所在行，内容B所在列交叉位置的值。
4. 结束数据获取原语句：如果当前获得的数据已经可以回答问题，请输出 [END]
    例：[END]

### Examples
{table_example}

{query_example}

### Query
{query}

### Table Content
{table}

### Other Query Answers
{query_history}

### Note
注意！输出的操作的每一部分都要使用[]括起来！
目的是获得可以回答问题的数据，而不是直接回答问题！你不需要输出额外的解释，不需要任何额外的标注引号，请你基于当前状态输出下一步要使用的工具，请你仅输出一步！
如果Table内容已经可以回答问题，请直接输出[END]结束！
"""

primtive_prompt_condition = """
### Instruction
现在有一个具有嵌套结构的表格，Table Content 给定了表格的具体数据，有一个针对该表格提出的问题，记作 ### Query。
现在请你根据问题，获得表格筛选子数据的条件，如果不需要针对问题对表格内容进行筛选，请直接输出None。
Examples是提供的样例，你的生成应该仿照这些样例的形式。
你可以用的条件比较符号有：
1. == :两者相等
2. >= :大于等于
3. <= :小于等于
4. > :大于
5. < :小于
6. != :不等于
7. in :一个字符串在另一个字符串中
你的输出格式需要符合: [COND] + [列名] + [操作] + [条件值]
例如查找表格中姓名为张三的数据: [COND] + [姓名] + [==] + [张三]
例如在商品列中查找值包含“铅笔”一词的数据: [COND] + [商品] + [in] + [铅笔]

### Query
{query}

### Table Content
{table}

### Note
注意！如果不需要针对问题对表格内的内容进行筛选，请直接输出None。
如果这个问题里包含多个条件，请分多行输出。例如查找银行里2016年一共存款了几次：
[COND] + [年份] + [==] + [2016]
[COND] + [操作] + [==] + [存款]
请你不要输出任何额外的解释，请直接输出操作语句。
"""

primitive_prompt_math = """
### Instruction
现在有一个具有嵌套结构的表格，Table Content 给定了表格的具体数据，有一个针对该表格提出的问题，记作 ### Query。
现在请你根据问题，分析是否需要对表格的数据进行数学操作，你可以使用的数学操作总共有五种，将在下文中给出。如果不需要进行数学操作，请直接输出None。
Examples是提供的样例，你的生成应该仿照这些样例的形式。
你可以用的数学操作有：
1. CNT：计数
2. SUM：求和
3. AVR：求平均值
4. MIN：求最小值
5. MAX：求最大值
你的输出格式需要符合: [MATH] + [列名] + [操作]
例如求所有男生的个数: [MATH] + [性别] + [CNT]
例如求信息箱对照表中有多少个条目：[MATH] + [条目] + [CNT]
例如求公司的产值总和: [MATH] + [产值] + [SUM]
例如求生产的产品的总成本：[MATH] + [成本] + [SUM]
例如求公司每月产值的平均值：[MATH] + [每月产值] + [AVR]
例如获取班级里年龄最小的同学的姓名：[MATH] + [年龄] + [MIN]
例如获取班级里年龄最大的同学的姓名：[MATH] + [年龄] + [MAX]
例如求成本最高的项目：[MATH] + [成本] + [MAX]

### Query
{query}

### Table Content
{table}

### Note
注意！如果回答这个问题不需要进行你可以使用的这五种数学操作，那么请直接输出None。
请你不要输出任何额外的解释，请直接输出操作语句。
"""

primitive_prompt_zeroshot = """
### Instruction
现在有一个具有嵌套结构的表格，有一个针对该表格提出的问题，记作 ### Query。
Tools是提供的原语句，请你使用这些原语句，一步一步的获取可供回答该问题的相关表格数据。你的原语句生成需要既考虑表格的结构，又考虑问题的形式。
Table Content 给定了表格a的具体数据。你可以通过生成原语句缩小寻找问题答案的范围。
Examples是提供的样例，包含表格，问题，和期望你生成的元语句，并且配有为何生成这个元语句的详细的说明。你的元语句生成应该仿照这些样例的形式。
Other Query Answers 记录了其他子问题的答案。

### Tools
1. 子节点提取原语句：提取某结点的所有后继节点，请输出 [CHL] + [KEY]
    例：[CHL] + [姓名]
2. 父节点提取原语句：提取某结点的父节点，请输出 [FAT] + [KEY]
    例：[FAT] + [张三]
3. 数据提取原语：根据行列提取某个位置的值，请输出 [EXT] + [ROW_KEY] + [COLUMN_KEY]
    例：[EXT] + [A] + [B] 获取内容A所在行，内容B所在列交叉位置的值。
4. 结束数据获取原语句：如果当前获得的数据已经可以回答问题，请输出 [END]
    例：[END]

### Query
{query}

### Table Content
{table}

### Other Query Answers
{query_history}

### Note
注意！输出的操作的每一部分都要使用[]括起来！
目的是获得可以回答问题的数据，而不是直接回答问题！你不需要输出额外的解释，不需要任何额外的标注引号，请你基于当前状态输出下一步要使用的工具，请你仅输出一步！
如果Table内容已经可以回答问题，请直接输出[END]结束！
"""

query_decompose_prompt = """
### Instruction
现在有一个具有嵌套结构的表格，有一个针对该表格提出的问题，请你将当前的问题分解为多个简单的子问题，并输出问题列表。\
注意，你仅需要拆解可以进行并列操作的问题，即带有"和","或","分别"等连接词的问题，或是少数几项求和作差的问题！
你还需要判断当前问题是否需要根据表格内容检索，还是可以根据前面几个子问题进行回答。
如果需要进行表格内容检索，输出True，否则输出False。

### Nested Schema
{schema}

### Query
{query}

### Example
Input: 2021年和2022年工资之和是多少？
Output:
[Query] 查询2021年工资。 [Retrieve] True
[Query] 查询2022年工资。 [Retrieve] True
[Query] 求2021年工资和2022年工资的和。 [Retrieve] False

Input: 2021年和2022年工资之差是多少？
Output:
[Query] 查询2021年工资。 [Retrieve] True
[Query] 查询2022年工资。 [Retrieve] True
[Query] 求2021年工资和2022年工资的差。 [Retrieve] False

Input: 张三的年龄是多少？
Output:
[Query] 张三的年龄是多少？ [Retrieve] True

Input: 社会与法学院中专科毕业生人数最少的专业是什么？
Output:
[Query] 社会与法学院中专科毕业生人数最少的专业是什么？ [Retrieve] True

Input: 管理费用中的水电费和折旧费中的办公设备折旧费分别属于什么类别？
Output:
[Query] 管理费用中的水电费属于什么类别？ [Retrieve] True
[Query] 折旧费中的办公设备折旧费属于什么类别？ [Retrieve] True
[Query] 管理费用中的水电费和折旧费中的办公设备折旧费分别属于什么类别？ [Retrieve] False

### Note
你不需要输出额外的解释，不需要任何额外的标注，一行输出一个子问题！
问题拆解可能并不是必要的，如果不需要拆解，请按规定格式输出原问题！
"""

entity_extract_prompt = """
### Instruction
我现在有一个表格和一个针对表格提问的问题，请将问题中的可能会用于表格内容检索的名词提取出来，返回一个python列表。

### Table Schema
{schema}

### Query
{query}

### Note
你不需要输出额外的解释，不需要任何额外的标注，仅需要输出一个python列表
"""


back_verification_prompt = """
### Instruction
现在给出了一个JSON格式的表格，一个基于表格问出的问题，以及可能的回答。
现在，请你基于这个表格和给出的回答，反向生成可能以给定答案作为回答的问题，并且生成{n}个。
请你每一行输出一个答案是{answer}的问题，共{n}个，并且不要任何额外的格式。

### Table
{table}

### Query
{query}

### Answer
{answer}
    
### Note
你仅需要一行一个的输出可能的问题，不需要标序号，也不需要其他任何格式。
"""

check_answer_prompt = """
### Instruction
现在给出了一个问题和大模型针对这个问题给出的答案，如果答案表达了以下类似的意思：“给出的数据不足，无法回答”，“无可供参考的数据”，“无法回答这个答案”，请直接输出F，否则输出T。

### Query
{query}

### Answer
{answer}

### Note
请直接输出T或F，不要有任何额外的格式。
"""

semantic_reasoning_prompt = """
### Instruction
现在有以下数据，请基于该数据，对下面的问题进行回答。

### Evidence
{evidence}

### Query
{query}

### Note
请不要有任何的的解释，不需要任何的标注，仅需要输出一个答案。
"""


direct_table_reasoning_prompt = """
### Instruction
现在有一个JSON表示的表格，请根据这个表格回答以下的问题。

### Table
{table}

### Query
{query}

### Note
请不要有任何额外的解释，请不要有任何的标注，请需要输出一个答案。
"""

evaluation_prompt = """
### Instruction
现在有两个值A，B，他们都是一个问题的回答，A是正确答案，B是其他人的回答，请你根据A判断B的回答是否正确，正确输出T，错误输出F。
如果A和B在数值上是相等的而单位不同，如498与498亿元，98与98人，也算作回答正确。

### Input
A: {a}
B: {b}

### Note
请直接输出T或F，T代表结果正确，F代表结果错误，你只要输出结果，不要任何额外的解释和格式！
"""