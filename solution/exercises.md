# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature = 0.0, phản hồi có tính nhất quán cao nhất, logic và ngắn gọn, lặp lại chính xác thông tin phổ biến (như bờ biển dài hay hang Sơn Đoòng). Khi tăng dần temperature lên 0.5 và 1.0, phản hồi đa dạng hơn về cấu trúc câu và từ ngữ, đưa ra các chi tiết phong phú hơn. Ở temperature = 1.5, câu từ trở nên rất ngẫu nhiên, sáng tạo nhưng có thể thiếu tự nhiên, đôi khi gặp lỗi ngữ pháp hoặc ảo giác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đối với chatbot hỗ trợ khách hàng, nên đặt temperature thấp từ 0.0 đến 0.2. Nguyên nhân là vì hỗ trợ khách hàng đòi hỏi tính chính xác tuyệt đối, thông tin đồng nhất và đáng tin cậy về chính sách, giá cả và dịch vụ, tránh việc model sáng tạo ngẫu nhiên hoặc đưa ra thông tin sai lệch (hallucination) gây ảnh hưởng xấu tới khách hàng và doanh nghiệp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o ($0.010/1k output tokens) đắt hơn GPT-4o-mini ($0.0006/1k output tokens) khoảng 16.67 lần cho phần output token.
> - **Trường hợp xứng đáng dùng GPT-4o:** Phân tích hợp đồng pháp lý phức tạp, chẩn đoán y khoa, hoặc giải quyết bài toán suy luận đa bước phức tạp mà độ chính xác quyết định sự thành bại.
> - **Trường hợp nên dùng GPT-4o-mini:** Chatbot phân loại phản hồi khách hàng đơn giản, tóm tắt email ngắn, hoặc tư vấn thông tin FAQ cơ bản với lưu lượng truy cập lớn nhằm tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi cho trẻ 8 tuổi dùng từ ngữ giản dị, ví dụ ẩn dụ trực quan (như cuốn sổ nhật ký dùng chung của lớp học), câu ngắn và giọng điệu thân thiện. Ngược lại, phản hồi cho chuyên gia tài chính dùng nhiều thuật ngữ chuyên ngành (sổ cái phân tán, mật mã phi tập trung, cơ chế đồng thuận, tính bất biến) và phân tích sâu về cấu trúc dữ liệu. System prompt đóng vai trò định hình vai trò (persona), điều chỉnh góc nhìn, văn phong, thuật ngữ và độ sâu kiến thức của model trong suốt quá trình tạo phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Khi đếm một đoạn tiếng Việt ~100 từ, `tiktoken` (tokenizer `o200k_base`/`cl100k_base`) trả về khoảng 130–160 token, trong khi ước lượng `100 / 0.75 ≈ 133` token. Số token thực tế thường cao hơn ước lượng từ ngữ tiếng Anh khoảng 15%–30%. Lý do là các bộ mã hóa BPE (Byte Pair Encoding) được huấn luyện chủ yếu trên dữ liệu tiếng Anh, do đó từ vựng tiếng Anh được mã hóa thành các subword/token nguyên vẹn, trong khi tiếng Việt có nhiều ký tự có dấu (Unicode UTF-8 đa byte) và ít từ ghép trong vocabulary nên hay bị tách thành nhiều fragment/subword nhỏ hơn, dẫn tới tốn nhiều token hơn trên cùng một dung lượng văn bản.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng như chatbot hội thoại hoặc trợ lý viết lách, nơi giúp giảm độ trễ cảm nhận (perceived latency) — người dùng thấy chữ xuất hiện ngay lập tức sau 1–2 giây thay vì phải chờ 10–20 giây cho phản hồi hoàn chỉnh. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm (background jobs), gọi API tự động giữa các hệ thống (system-to-system integration), phân tích dữ liệu dạng JSON/structured output hoặc các pipeline cần validation hoàn chỉnh văn bản trước khi xử lý bước tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm áp lực tức thì lên hệ thống server bằng cách dãn cách thời gian chờ tăng gấp đôi giữa các lần thử lại. Nếu hàng nghìn client cùng gặp lỗi và retry với delay cố định giống nhau (như 1 giây), tất cả sẽ đồng loạt gửi lại request tại đúng thời điểm 1s, tạo ra hiện tượng "thảm họa dồn dập" (retry storm / thundering herd problem) làm server đã nghẽn càng thêm quá tải và crash hoàn toàn. Exponential backoff giải tán lượng request phân bổ rải rác theo thời gian, giúp server có khoảng thở để phục hồi.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
> 解釋 (Giải thích lựa chọn từ ngữ):
> 1. **"ngắn gọn"**: Giúp model giới hạn độ dài phản hồi, đi thẳng vào trọng tâm câu hỏi, tiết kiệm token input/output và giảm chi phí API.
> 2. **"bằng tiếng Việt"**: Đảm bảo model luôn giữ ngôn ngữ phản hồi thống nhất cho học viên Việt Nam ngay cả khi câu hỏi đầu vào chứa thuật ngữ tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại là lịch sử hội thoại chỉ lưu tối đa 3 lượt (6 messages) nên trợ lý sẽ hoàn toàn "quên" các thông tin đã trao đổi từ lượt thứ 4 trở về trước, đồng thời chưa có khả năng truy xuất kiến thức bên ngoài (RAG) hay lưu trữ bộ nhớ dài hạn.
> **Đề xuất cải thiện:** Triển khai kỹ thuật Tóm tắt lịch sử (Conversation Summary). Cụ thể: khi history vượt quá 6 messages, sử dụng một LLM call nhỏ (như `gpt-4o-mini`) để tóm tắt các lượt hội thoại cũ thành một đoạn tóm tắt ngắn (summary context) và đính kèm vào system prompt, giúp trợ lý duy trì bộ nhớ ngữ cảnh dài hạn mà không làm bùng nổ số lượng token.

---

## Danh Sách Kiểm Tra Nộp BÀI

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
