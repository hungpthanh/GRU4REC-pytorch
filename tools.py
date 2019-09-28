if __name__ == '__main__':
    file = "data/raw_data/yoochoose-clicks.dat"
    file_out = "data/raw_data/yoochoose-clicks-super-small.dat"
    content = []
    with open(file, 'r') as f:
        for line in f:
            content.append(line)
    print(len(content))

    small_index = len(content) // 8
    #small_index = 100
    with open(file_out, 'w') as f:
        for line in content[-small_index:]:
            f.write(line)