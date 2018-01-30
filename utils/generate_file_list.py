import os

if __name__ == '__main__':
	
	# main()
	split_num = 1
	partition = 'test'
	ori_file_list_root = './data/ucf101/train_test_lists/action_back'
	ori_file_list_path = ori_file_list_root+'/'+partition+'_split'+str(split_num)+'.txt'
	print(ori_file_list_path)
	#output path
	output_path = './data/ucf101/train_test_lists'+'/'+partition+'_split'+str(split_num)+'.txt'

	img_root = './data/ucf101/frames'
	wf = open(output_path,'w')
	with open(ori_file_list_path,'r') as rf:
		for idx, line in enumerate(rf):
			items = line.strip().split(' ')
			imgs = os.listdir(img_root+'/'+items[0])
			wf.write('%s %d %s\n' %(items[0], len(imgs), items[2]))

	wf.close()
