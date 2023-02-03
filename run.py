from MMEdu import MMPose # 导入mmpose模块
import shutil
import os
import json
import multiprocessing

model = MMPose(backbone='SCNet') # 实例化mmpose模型
pose_model, det_model = model.init_model(device="cpu")

origin_list = os.listdir("./imgs/origin/1")
finish_list = os.listdir("./imgs/output/1")
for finish_file in finish_list:
    if ".json" in finish_file:
        continue
    else:
        origin_list.remove(finish_file)

def img_process(img_name, tags):
    output_path = "./imgs/output/1/"
    img_path = f"./imgs/origin/1/{img_name}"
    result = model.inference(img=img_path, device='cpu', show=False, name=f'pose_result{tags}', pose_model=pose_model, det_model=det_model) # 在CPU上进行推理
    shutil.move(f"./pose_result{tags}.png", output_path)
    os.rename(f"./imgs/output/1/pose_result{tags}.png", f"./imgs/output/1/{img_name}")
    result_list = []
    for index,people in enumerate(result):
        keypoints = people['keypoints'][:, :].tolist()
        bbox = people['bbox'][:].tolist()
        result_list += [{"bbox": bbox, "keypoints": keypoints}]
    with open(f"./imgs/output/1/{img_name}".replace(".png", ".json"), "w", encoding="utf-8") as f:
        json.dump(result_list, fp=f, indent=2, ensure_ascii=False)

def img_to_process(total, this):
    for idx, file in enumerate(origin_list):
        if os.path.isdir(f"./imgs/origin/1/{file}"):
            continue
        else:
            if idx % total == this:
                img_process(file, this)
            else:
                continue

if __name__ == "__main__":      
    process_pool = []
    total_process = 6
    for prc_num in range(total_process):
        this_process = multiprocessing.Process(target=img_to_process, args=(total_process, prc_num))
        this_process.start()
        process_pool.append(this_process)
    for this_process in process_pool:
        this_process.join()
    print("执行结束")
