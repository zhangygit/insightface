import mxnet as mx
import os
import argparse

def make_perfect_insightface_rec(root_dir, save_dir):
  lst_path = os.path.join(save_dir, 'train.lst')
  rec_path = os.path.join(save_dir, 'train.rec')
  idx_path = os.path.join(save_dir, 'train.idx')

  img_list = []
  with open(lst_path, 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('\t')
      # 记录：原始标签, 图片相对路径
      img_list.append((int(parts[0]), float(parts[1]), parts[2]))

  record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

  # 3. 【关键】写入索引 0 的元数据头
  # InsightFace 习惯：label[0] = 最大索引 + 1 (即 count + 1)
  # label[1] = flag (通常设为 2)
  max_idx = max([x[0] for x in img_list])
  header0 = mx.recordio.IRHeader(0, [float(max_idx + 1), float(max_idx + 1)], 0, 0)
  record.write_idx(0, mx.recordio.pack(header0, b''))

  # 4. 写入图片数据
  print(f"Total images: {len(img_list)}. Starting conversion...")
  for i, label, rel_path in img_list:
    idx = i +1
    full_path = os.path.join(root_dir, rel_path)
    with open(full_path, 'rb') as f:
      img_data = f.read()

    # 这里的 label 也建议包装成列表 [label]
    header = mx.recordio.IRHeader(0, label, idx, 0)
    s = mx.recordio.pack(header, img_data)
    record.write_idx(idx, s)

    if i % 1000 == 0:
      print(f"Processed {i}...")



  record.close()
  print(f"转换完成！文件保存至: {save_dir}")

if __name__ == "__main__":
  # 命令行参数解析
  parser = argparse.ArgumentParser(description="将 InsightFace 格式的 .lst 转换为 .rec 和 .idx 文件")

  parser.add_argument("--root", type=str, required=True,
                      help="图片所在的根目录 (image_folder 的路径)")
  parser.add_argument("--save", type=str, required=True,
                      help="train.lst 所在目录，也是 rec/idx 的生成目录")

  args = parser.parse_args()

  # 执行转换
  make_perfect_insightface_rec(args.root, args.save)