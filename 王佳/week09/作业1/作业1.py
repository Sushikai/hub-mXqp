def main(sex: str, height: int, weight: int, age: int) -> dict:
    # 根据性别选择不同的体脂率计算公式
    if sex.lower() == '男':
        # 男性体脂率计算公式
        body_fat_percentage = 1.2 * (weight / (height / 100) ** 2) + 0.23 * age - 16.2
    elif sex.lower() == '女':
        # 女性体脂率计算公式
        body_fat_percentage = 1.2 * (weight / (height / 100) ** 2) + 0.23 * age - 5.4

    # 确保体脂率在合理范围内
    if body_fat_percentage < 0:
        body_fat_percentage = 0
    elif body_fat_percentage > 100:
        body_fat_percentage = 100

    # 根据不同年龄段和性别判断体脂等级
    health_level, advice = get_health_advice(body_fat_percentage, age, sex)

    return {
        "body_fat_percentage": round(body_fat_percentage, 2),
        "health_level": health_level,
        "advice": advice
    }

def get_health_advice(body_fat: float, age: int, sex: str) -> tuple:
    """
    根据体脂率、年龄和性别提供健康建议
    """
    # 不同年龄段的正常体脂范围
    if age < 30:
        if sex == '男':
            excellent = (6, 14)
            normal = (8, 20)
        else:
            excellent = (14, 20)
            normal = (16, 24)
    elif age < 50:
        if sex == '男':
            excellent = (8, 16)
            normal = (10, 22)
        else:
            excellent = (16, 22)
            normal = (18, 26)
    else:
        if sex == '男':
            excellent = (10, 18)
            normal = (12, 25)
        else:
            excellent = (18, 24)
            normal = (20, 28)

    # 判断体脂等级并生成建议
    if body_fat < excellent[0]:
        level = "过低"
        advice = "您的体脂率非常低，建议适当增加营养摄入，结合力量训练增肌。"
    elif body_fat < normal[0]:
        level = "优秀"
        advice = "体脂率很棒！保持当前的饮食和运动习惯即可。"
    elif body_fat <= normal[1]:
        level = "正常"
        advice = "体脂率在正常范围内，继续保持健康的生活方式。"
    elif body_fat <= normal[1] + 5:
        level = "偏高"
        advice = "体脂率略高，建议增加有氧运动，控制高热量食物，每周减重 0.5-1kg。"
    else:
        level = "过高"
        advice = "体脂率明显超标，建议制定科学减脂计划，咨询专业健身教练或营养师。"

        # 根据年龄添加特别提示
    if age < 18:
        advice += "\n【青少年提示】身体还在发育中，不要过度节食，保证充足营养和睡眠。"
    elif age >= 50:
        advice += "\n【中老年提示】新陈代谢减慢，建议增加力量训练防止肌肉流失，注意补充钙质。"

    return level, advice

print(main("男", 175, 75, 38 ))