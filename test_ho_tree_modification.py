"""
测试HOtree修改功能的脚本
"""
import os
import json
import pickle
from table2tree.feature_tree import FeatureTree
from utils.data_type_utils import json_to_feature_tree

def test_json_to_feature_tree_conversion():
    """测试JSON到FeatureTree的转换功能"""
    print("开始测试HOtree修改功能...")
    
    # 创建一个简单的FeatureTree对象用于测试
    original_tree = FeatureTree()
    original_tree.name = "测试树"
    original_tree.value = "测试值"
    original_tree.structure_type = "kv"
    
    # 转换为JSON
    tree_json = original_tree.__json__()
    print("原始HOtree JSON:")
    print(json.dumps(tree_json, indent=2, ensure_ascii=False))
    
    # 修改JSON内容
    if isinstance(tree_json, dict):
        tree_json['value'] = "修改后的值"
        tree_json['name'] = "修改后树"
    
    print("\n修改后的HOtree JSON:")
    print(json.dumps(tree_json, indent=2, ensure_ascii=False))
    
    # 转换回FeatureTree
    try:
        converted_tree = json_to_feature_tree(tree_json)
        print("\n转换成功！FeatureTree对象属性:")
        print(f"  - name: {converted_tree.name}")
        print(f"  - value: {converted_tree.value}")
        print(f"  - structure_type: {converted_tree.structure_type}")
        print("\nHOtree修改功能测试通过！")
        return True
    except Exception as e:
        print(f"\n转换失败: {str(e)}")
        return False

def test_cache_mechanism():
    """测试缓存机制"""
    print("\n测试HOtree缓存机制...")
    
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 创建测试JSON文件
    test_json = {
        "name": "缓存测试树",
        "value": "缓存测试值",
        "structure_type": "kv"
    }
    
    json_path = os.path.join(cache_dir, "temp.modified.json")
    pkl_path = os.path.join(cache_dir, "temp.modified.pkl")
    
    # 保存JSON
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(test_json, f, indent=4, ensure_ascii=False)
    
    # 测试JSON转FeatureTree并保存为pickle
    tree = json_to_feature_tree(test_json)
    with open(pkl_path, "wb") as f:
        pickle.dump(tree, f)
    
    # 从pickle加载
    with open(pkl_path, "rb") as f:
        loaded_tree = pickle.load(f)
    
    print(f"缓存测试通过！从pickle加载的树: {loaded_tree.name} - {loaded_tree.value}")
    return True

if __name__ == "__main__":
    try:
        test1_passed = test_json_to_feature_tree_conversion()
        test2_passed = test_cache_mechanism()
        
        if test1_passed and test2_passed:
            print("\n🎉 所有测试通过！HOtree修改功能已准备就绪！")
            print("\n使用说明:")
            print("1. 启动gradio_app.py")
            print("2. 上传Excel表格")
            print("3. 在H-OTree JSON组件中直接编辑结构")
            print("4. 点击'保存HOtree修改'按钮")
            print("5. 后续的问答将使用修改后的HOtree结构")
        else:
            print("\n❌ 部分测试失败，请检查代码")
    except Exception as e:
        print(f"测试运行出错: {str(e)}")
