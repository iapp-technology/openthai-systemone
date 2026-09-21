from openthai_systemone import SystemOneClient, Choice, Score, Noul

client = SystemOneClient("iapp/OpenThai-SystemOne")
resp = client.system_one(
    state={"ticket": "ลูกค้าแจ้งว่าโดนหักเงินซ้ำสองครั้ง ขอเงินคืนด่วน โทรมาสามรอบแล้ว"},
    questions={
        "department": Choice(instructions="ทีมใดควรรับผิดชอบ", criteria={"billing": "การเงิน/ค่าบริการ", "technical": "ระบบใช้งานไม่ได้", "sales": None}),
        "frustration": Score(instructions="ลูกค้าหงุดหงิดแค่ไหน", criteria=["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]),
        "refund_requested": Noul(instructions="ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"),
    },
)
print(resp.answers["department"].choice, resp.answers["department"].probabilities)
print(resp.answers["frustration"].score, resp.answers["refund_requested"].noul)
