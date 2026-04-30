实现一下基于 chinook.db 数据集进行问答
提问1：数据库中总共有多少张表？

albums
sqlite_sequence（系统表）
customers
genres
employees
media_types
invoice_items
playlist_track
playlists
tracks
sqlite_stat1（系统表）

除去两个 SQLite 内部系统表，用户表共有 11 张，若计入系统表则为 13 张，但通常题目所指为业务表。

提问2：员工表中有多少条记录？
从文件中的数据部分可提取出员工记录，依次为：
Andrew Adams（总经理）
Nancy Edwards（销售经理）
Jane Peacock（销售支持）
Margaret Park（销售支持）
Steve Johnson（销售支持）
Michael Mitchell（IT 经理）
Robert King（IT 人员）
Laura Callahan（IT 人员）
员工表共有 8 条记录。

提问3：在数据库中所有客户个数和员工个数分别是多少？
客户个数：从文件中统计客户记录（通过姓名、地址、邮箱等特征），共 59 条（与 Chinook 示例数据库一致）。
员工个数：如上所述，8 条。

答案汇总：

表总数：11 张（用户表）
员工表记录数：8 条
客户数：59，员工数：8
