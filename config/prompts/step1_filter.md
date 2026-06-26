# 本地模型过滤指令
当前主题：**{theme}**

请为以下每个文本块输出两个标签：
1. **relevance**（相关性）：
   - `direct`：直接讲方法/案例/步骤。
   - `inspirational`：仅提供心法或启发。
   - `unrelated`：完全不相关（噪声、版权、闲聊等）。
2. **rough_density**（粗信息密度）：
   - 仅对 `direct` 和 `inspirational` 的块输出。
   - `high`：包含大量实体、数字或明显逻辑词。
   - `medium`：包含少量实体或逻辑。
   - `low`：几乎没有实体或数字（空泛、纯抒情）。

## 输出格式
每行一个块，格式：
```
[id]: relevance=direct, rough_density=high
```
只输出标签行，不要任何解释。

---

## 待处理块