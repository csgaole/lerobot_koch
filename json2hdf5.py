for episode_key, episode in tqdm(dataset.items(), desc="Converting dataset"):  # 使用 tqdm 显示进度条
    episode_group = hdf5_file.create_group(episode_key)
    
    # 存储语言指令
    if 'language_instruction' in episode:
        episode_group.create_dataset('language_instruction', data=episode['language_instruction'])
    
    # 存储状态数据
    if 'states' in episode:
        states = np.array(episode['states'])
        episode_group.create_dataset('states', data=states, dtype=np.float32)
    
    # 存储动作数据
    if 'actions' in episode:
        actions = np.array(episode['actions'])
        episode_group.create_dataset('actions', data=actions, dtype=np.float32)
    
    # 存储图像数据 (如果存在)
    if 'images' in episode:
        images = np.array(episode['images'])
        episode_group.create_dataset('images', data=images, dtype=np.uint8)
    
    # ... 存储其他数据 (根据您的数据集结构)