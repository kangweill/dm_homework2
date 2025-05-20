#/data/kw/datamining/
import os
import json
import pandas as pd
import pyarrow.parquet as pq
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict

# 设置字体
rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# -------------------- 配置参数 --------------------
CONFIG = {
    "data_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/10G_data_new")),
    "catalog_path": os.path.abspath("./product_catalog.json"),
    "output_dir": os.path.abspath("./results/"),
    "task1": {"min_support": 0.02, "min_confidence": 0.5},
    "task2": {"min_support": 0.01, "min_confidence": 0.6},
    "task4": {"min_support": 0.005, "min_confidence": 0.4},
    "batch_size": 1000,  # 设置较小的批次大小
    "high_value_threshold": 5000
}

# -------------------- 大类映射 --------------------
MAJOR_CATEGORIES = {
    "电子产品": ["智能手机", "笔记本电脑", "平板电脑", "智能手表", "耳机", "音响", "相机", "摄像机", "游戏机"],
    "服装": ["上衣", "裤子", "裙子", "内衣", "鞋子", "帽子", "手套", "围巾", "外套"],
    "食品": ["零食", "饮料", "调味品", "米面", "水产", "肉类", "蛋奶", "水果", "蔬菜"],
    "家居": ["家具", "床上用品", "厨具", "卫浴用品"],
    "办公": ["文具", "办公用品"],
    "运动户外": ["健身器材", "户外装备"],
    "玩具": ["玩具", "模型", "益智玩具"],
    "母婴": ["婴儿用品", "儿童课外读物"],
    "汽车用品": ["车载电子", "汽车装饰"]
}

# -------------------- 环境设置 --------------------

def setup_environment():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    if not os.path.exists(CONFIG['data_dir']):
        raise FileNotFoundError(f"数据目录 {CONFIG['data_dir']} 不存在")
    if not os.path.exists(CONFIG['catalog_path']):
        raise FileNotFoundError(f"商品目录文件 {CONFIG['catalog_path']} 不存在")

# -------------------- 加载目录 --------------------

def load_catalog():
    with open(CONFIG['catalog_path'], "r", encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_map = {str(p['id']): p.get('category', '未知') for p in catalog.get('products', [])}
    major_map = {sub: major for major, subs in MAJOR_CATEGORIES.items() for sub in subs}
    return catalog_map, major_map

# -------------------- 工具函数 --------------------

def extract_categories(record, catalog_map, major_map):
    cats = []
    try:
        data = json.loads(record)
        for item in data.get('items', []):
            cid = str(item.get('id', ''))
            sub = catalog_map.get(cid)
            if sub:
                cats.append(sub)
    except Exception:
        pass
    return list(set([major_map.get(c, '未知') for c in cats]))

# 判断高价值订单
def is_high_value(record):
    try:
        data = json.loads(record)
        # 平均价格字段改为 avg_price
        avg = data.get('avg_price')
        if isinstance(avg, (int, float)) and avg > CONFIG['high_value_threshold']:
            return True
        # 单品价格
        for item in data.get('items', []):
            price = item.get('price')
            if isinstance(price, (int, float)) and price > CONFIG['high_value_threshold']:
                return True
    except Exception:
        pass
    return False

# -------------------- 分析任务 --------------------

def analyze_task1(transactions):
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    freq = apriori(df, min_support=CONFIG['task1']['min_support'], use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=CONFIG['task1']['min_confidence'])
    elec = rules[rules['antecedents'].apply(lambda s: '电子产品' in s) | rules['consequents'].apply(lambda s: '电子产品' in s)]
    freq.to_csv(os.path.join(CONFIG['output_dir'], 'task1_freq.csv'), index=False)
    rules.to_csv(os.path.join(CONFIG['output_dir'], 'task1_rules.csv'), index=False)
    elec.to_csv(os.path.join(CONFIG['output_dir'], 'task1_elec.csv'), index=False)
    plt.figure(); freq['support'].hist(); plt.title('Task1 支持度分布'); plt.savefig(os.path.join(CONFIG['output_dir'], 'task1_support.png')); plt.close()
    return freq, rules, elec


def analyze_task2(transactions, hv_flags):
    # 全部订单 支付方式+类别
    recs = [cats + [pm] for pm, cats in transactions]
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(recs), columns=te.columns_)
    freq = apriori(df, min_support=CONFIG['task2']['min_support'], use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=CONFIG['task2']['min_confidence'])
    freq.to_csv(os.path.join(CONFIG['output_dir'], 'task2_freq.csv'), index=False)
    rules.to_csv(os.path.join(CONFIG['output_dir'], 'task2_rules.csv'), index=False)
    # 高价值订单首选支付方式
    hv = pd.Series([pm for (pm,_), hv in zip(transactions, hv_flags) if hv])
    hv_counts = hv.value_counts()
    hv_counts.to_csv(os.path.join(CONFIG['output_dir'], 'task2_hv_counts.csv'))
    plt.figure(); hv_counts.plot(kind='bar'); plt.title('高价值支付方式'); plt.savefig(os.path.join(CONFIG['output_dir'], 'task2_hv.png')); plt.close()
    return freq, rules, hv_counts


def analyze_task3(cnts, trans):
    for period in ['month', 'quarter', 'weekday']:
        dfm = pd.DataFrame(cnts[period]).fillna(0).astype(int)
        dfm.to_csv(os.path.join(CONFIG['output_dir'], f'task3_{period}.csv'))
        top3 = dfm.sum().nlargest(3).index
        plt.figure()
        dfm[top3].plot()
        plt.title(f'{period}前三')
        plt.savefig(os.path.join(CONFIG['output_dir'], f'task3_{period}.png'))
        plt.close()
    
    # 转移矩阵
    tdf = pd.DataFrame(trans).fillna(0).astype(int)
    tdf.to_csv(os.path.join(CONFIG['output_dir'], 'task3_trans.csv'))
    
    # 前20序列模式
    seq = [(a, b, c) for a, row in trans.items() for b, c in row.items()]
    sq = pd.DataFrame(seq, columns=['A', 'B', 'count'])
    sq['count'] = pd.to_numeric(sq['count'], errors='coerce')  # 确保 count 列为数值类型
    sq = sq.nlargest(20, 'count')
    sq.to_csv(os.path.join(CONFIG['output_dir'], 'task3_seq.csv'), index=False)
    return dfm, tdf, sq


def analyze_task4(transactions):
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
    freq = apriori(df, min_support=CONFIG['task4']['min_support'], use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=CONFIG['task4']['min_confidence'])
    freq.to_csv(os.path.join(CONFIG['output_dir'], 'task4_freq.csv'), index=False)
    rules.to_csv(os.path.join(CONFIG['output_dir'], 'task4_rules.csv'), index=False)
    plt.figure(); rules['lift'].hist(); plt.title('Task4 提升度'); plt.savefig(os.path.join(CONFIG['output_dir'], 'task4_lift.png')); plt.close()
    return freq, rules

# -------------------- 主流程 --------------------
def main():
    setup_environment()
    cmap, mmap = load_catalog()
    t1, t2, hv = [], [], []
    t3_cnt = {'month': defaultdict(lambda: defaultdict(int)), 'quarter': defaultdict(lambda: defaultdict(int)), 'weekday': defaultdict(lambda: defaultdict(int))}
    t3_trans = defaultdict(lambda: defaultdict(int))
    t4 = []
    
    # 处理所有文件
    files = [f for f in os.listdir(CONFIG['data_dir']) if f.endswith('.parquet')]
    for fn in files:
        # 读取文件的所有批次
        for batch in pq.ParquetFile(os.path.join(CONFIG['data_dir'], fn)).iter_batches(CONFIG['batch_size']):
            df = batch.to_pandas()
            if df.empty:
                continue
            df['cats'] = df['purchase_history'].apply(lambda x: extract_categories(x, cmap, mmap))
            ph = df['purchase_history'].apply(json.loads)
            df['pm'] = ph.apply(lambda d: d.get('payment_method'))
            df['ps'] = ph.apply(lambda d: d.get('payment_status'))
            df['hv'] = df['purchase_history'].apply(is_high_value)
            df['ptime'] = pd.to_datetime(ph.apply(lambda d: d.get('purchase_date')), errors='coerce')
            df['month'] = df['ptime'].dt.month.fillna(0).astype(int)
            df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
            df['weekday'] = df['ptime'].dt.weekday.fillna(-1).astype(int)
            
            # 收集数据
            t1.extend(df['cats'])
            for _, r in df.iterrows():
                if r['pm'] and r['cats']:
                    t2.append((r['pm'], r['cats']))
                    hv.append(r['hv'])
            for _, r in df.iterrows():
                for c in r['cats']:
                    t3_cnt['month'][r['month']][c] += 1
                    t3_cnt['quarter'][r['quarter']][c] += 1
                    t3_cnt['weekday'][r['weekday']][c] += 1
            uid = 'user_id' if 'user_id' in df.columns else 'id'
            for uid_val, grp in df.groupby(uid):
                seqs = grp.sort_values('ptime')['cats'].tolist()
                for i in range(len(seqs) - 1):
                    for a in seqs[i]:
                        for b in seqs[i + 1]:
                            if a != b:
                                t3_trans[a][b] += 1
            t4.extend(df[df['ps'].isin(['已退款', '部分退款'])]['cats'])
    
    analyze_task1(t1)
    analyze_task2(t2, hv)
    analyze_task3(t3_cnt, t3_trans)
    analyze_task4(t4)
    print("完成，结果在", CONFIG['output_dir'])

if __name__ == '__main__':
    main()

