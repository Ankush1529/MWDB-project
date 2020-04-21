import sys
sys.path.insert(1, '../Phase1')
sys.path.insert(1, '../Phase2')
sys.path.insert(1, '../Phase3')

import itertools as it
from Decomposition import Decomposition
import misc, os
import pandas as pd
import numpy as np
import heapq
from page_rank import k_neighbour_graph, steady_state, norm_distance
from dtree import *
import test
from basicsvm import SVMClassifier as svmc

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def get_labels(img_list, df):
    class_labels = []
    for img in img_list: 
        class_list = df['aspectOfHand'].where(df['imageName']==img)
        class_list = [class_l for class_l in class_list if str(class_l) != 'nan']
        # print(str(class_list[0].split()[0]) + img )
        if(class_list[0].split()[0] == "palmar"):
            class_labels.append(1) 
        if(class_list[0].split()[0] == "dorsal"):
            class_labels.append(0) 
        
    return class_labels 

    
labelled_set_path = input("Enter the path for labelled input set - ")
unlabelled_set_path = input("Enter the path for unlabelled input set - ")
metadata_file_path = input("Enter the path for metadata csv - ")
metadata_file_name = input("Enter the name for labelled metadata csv - ")
unlabelled_metadata_file_name = metadata_file_name

# unlabelled_metadata_file_name = input("Enter the name for unlabelled metadata csv - ")
metadata_df = pd.read_csv(metadata_file_path+'/'+metadata_file_name)
unlabelled_metadata_df = pd.read_csv(metadata_file_path+'/'+unlabelled_metadata_file_name)

# print(metadata_df)
unlabelled_images_list = list(misc.get_images_in_directory(unlabelled_set_path).keys())



reduced_pickle_file_folder =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pickle_files')
classifier = input("Choose the Classifier - \n. 1. SVM \n. 2. Decision Tree \n. 3. PPR Classifier \n. 4. Exit \n. ")


def svm():
    model = "HOG"
    decomposition_model = "PCA"
    phase = input("Choose from \n. 1. Train \n. 2. Test \n.")
    if (phase == "train"):
        # phase = "train"
        training_features,images_list = test.compute_features_folder(labelled_set_path, phase)
        
        metadata_filepath = metadata_file_path + "/" + metadata_file_name
        
        csv_labels=test.make_labels(metadata_filepath)
        binary_labels=[]
        for i in csv_labels:
            if "dorsal" in i:
                binary_labels.append(0)
            else:
                binary_labels.append(1)
                
        my_model = svmc(.001, .01, 1000)
        
        my_model.fit(training_features,binary_labels)
        misc.save2pickle(my_model, reduced_pickle_file_folder,feature = (model + '_svm'))
        
    if(phase == "test"):    
        # phase = "test"
        testing_features,images_list = test.compute_features_folder(unlabelled_set_path, phase)
        
        my_model = misc.load_from_pickle(reduced_pickle_file_folder, feature = (model + '_svm'))
        value=[]
        
        value = my_model.predict(testing_features)
        ans = []
        for i in value:
            if(i == -1):
                ans.append("dorsal")
            else:
                ans.append("palmar")
        svm_dict={}
        i=0
        for img in images_list:
            svm_dict[img]=ans[i]
            i+=1
        print("HI")
        print(svm_dict)
        # test.accuracy(svm_dict)
        
def decision_tree_input():
    model = "CM"
    decomposition_model = "NMF"
    k = 8
    phase = input("Choose from \n. 1. Train \n. 2. Test \n.")
    if (phase == "train"):
        # phase = "train"
        decomposition = Decomposition(decomposition_model, k, model, labelled_set_path, phase)
        decomposition.dimensionality_reduction()
        
        labelled_images_feature_dict = misc.load_from_pickle(r"D:\MS_1\MWDB-project\Phase2\pickle_files",feature = (model + '_' + decomposition_model + '_train'))
        
        y_train = get_labels(labelled_images_feature_dict.keys(),metadata_df)
        X_train = np.vstack(labelled_images_feature_dict.values())
        y_train = np.asarray(y_train).reshape(len(y_train),1)
        
        dataset = np.concatenate((X_train,y_train)  ,axis=1)

        tree = build_tree(dataset, 6, 1)
        
        misc.save2pickle(tree, reduced_pickle_file_folder,feature = (model + '_' + decomposition_model + '_tree'))
        
    if(phase == "test"):    
        # phase = "test"
        tree = misc.load_from_pickle(reduced_pickle_file_folder,feature = (model + '_' + decomposition_model + '_tree'))

        decomposition = Decomposition(decomposition_model, k, model, unlabelled_set_path, phase)
        decomposition.dimensionality_reduction()

        
        unlabelled_images_feature_dict = misc.load_from_pickle("D:\MS_1\MWDB-project\Phase2\pickle_files",feature = (model + '_' + decomposition_model + '_test'))

        y_test = get_labels(unlabelled_images_feature_dict.keys(),unlabelled_metadata_df)

        X_test = np.vstack(unlabelled_images_feature_dict.values())

        # print(y_train.shape)
        # print(y_test)
        
        # print(tree)
        prediction = {}
        for key in unlabelled_images_feature_dict.keys():
            est_val = predict(tree, unlabelled_images_feature_dict[key])
            if (est_val == 1):
                prediction[key] = "palmar"
            else:
                prediction[key] = "dorsal"
            
        print(prediction)
        
        correct = 0
        class_list = unlabelled_metadata_df['imageName'].tolist()
        actual_class_list = unlabelled_metadata_df['aspectOfHand'].tolist()
        # print(actual_class_list)
        for image_name in prediction.keys():
            class_list = unlabelled_metadata_df['aspectOfHand'].where(unlabelled_metadata_df['imageName']==image_name)
                    
            class_list = [class_l for class_l in class_list if str(class_l) != 'nan']
            # print(str(class_list[0].split()[0]) + "--" + image_name )
            if(class_list[0].split()[0] == prediction.get(image_name)):
                correct += 1
        print(correct/len(prediction.keys()))
       


def ppr_classifier():
    model = "CM"
    decomposition_model = "NMF"
    k = 8
    phase = input("Choose from \n. 1. Train \n. 2. Test \n.")
    if (phase == "train"):
        
        decomposition = Decomposition(decomposition_model, k, model, labelled_set_path, phase)
        decomposition.dimensionality_reduction()    
        reduced_dim_folder_images_dict = misc.load_from_pickle("D:\MS_1\MWDB-project\Phase2\pickle_files",feature = (model + '_' + decomposition_model + '_' + phase))
        image_image_graph_keys = ()
        columns = list(reduced_dim_folder_images_dict.keys())
        image_image_graph_keys = list(it.combinations(reduced_dim_folder_images_dict.keys(),2))
        image_image_df = pd.DataFrame(0.00, columns = reduced_dim_folder_images_dict.keys(), index = reduced_dim_folder_images_dict.keys())
        image_image_df_top_features = pd.DataFrame(0.00, columns = reduced_dim_folder_images_dict.keys(), index = reduced_dim_folder_images_dict.keys())
        
        image_image_df = norm_distance(image_image_df, reduced_dim_folder_images_dict, image_image_graph_keys, 2)
        misc.save2pickle(image_image_df, reduced_pickle_file_folder, feature=(model + '_' + decomposition_model + '_image_image_df'  ))
    
    
    if (phase == "test"):    
        phase = "test"
        decomposition = Decomposition(decomposition_model, k, model, unlabelled_set_path, phase)
        decomposition.dimensionality_reduction()
        labelled_images_feature_dict = misc.load_from_pickle("D:\MS_1\MWDB-project\Phase2\pickle_files",feature = (model + '_' + decomposition_model + '_train'))
        unlabelled_images_feature_dict = misc.load_from_pickle("D:\MS_1\MWDB-project\Phase2\pickle_files",feature = (model + '_' + decomposition_model + '_test'))


        # K = int(input("Enter the number of dominant images - "))
        K = 9
        prediction = {}

        image_image_df = misc.load_from_pickle(reduced_pickle_file_folder,feature = (model + '_' + decomposition_model + "_image_image_df"))

           
        for unlabelled_img in unlabelled_images_list:
            # unlabelled_img = "Hand_0000070.jpg"
            # new_col_dict = {}
            new_col = []
            
            for labelled_img in labelled_images_feature_dict.keys():
                features1 = unlabelled_images_feature_dict.get(unlabelled_img)
                features2 = labelled_images_feature_dict.get(labelled_img)
                ind_distance = 0.00
                distance = 0.00
                # print(features1)
                # print("--------------")
                # print(features2)
                for i in range(len(features1)):
                    ind_distance = abs(features1[i] - features2[i])
                    distance += (ind_distance ** 2)
                    
                distance = distance ** (1/float(2))
                # new_col_dict[labelled_img] = distance
                new_col.append(distance)
            # print(new_col)
            # new_row = pd.DataFrame(new_col, columns=image_image_df.columns, index=[unlabelled_img])

            # image_image_df = image_image_df.append(new_row)
            image_image_df = image_image_df.append(pd.Series(new_col, index=image_image_df.columns, name=unlabelled_img))
            # image_image_df = pd.concat([image_image_df, new_row_df])
            new_col.append(0)
            # print(new_col)
            image_image_df = image_image_df.assign(unlabelled_img = new_col)
            image_image_df = image_image_df.rename({'unlabelled_img' : unlabelled_img},axis = 1)

            image_image_df = image_image_df.loc[:,~image_image_df.columns.duplicated()]
            image_image_df = image_image_df[~image_image_df.index.duplicated(keep='first')]
            image_image_features_df = k_neighbour_graph(image_image_df, image_image_df.columns, 8)
            dominant_img_list = steady_state([unlabelled_img],image_image_features_df, image_image_features_df.columns, K)

            # print(dominant_img_list)
            palmar = 0
            dorsal = 0
            for img in dominant_img_list: 
                if img not in unlabelled_img:
                    class_list = metadata_df['aspectOfHand'].where(metadata_df['imageName']==img)
                    class_list = [class_l for class_l in class_list if str(class_l) != 'nan']
                    # print(str(class_list) + img )
                    if(class_list[0].split()[0] == "palmar"):
                        palmar += 1
                    if(class_list[0].split()[0] == "dorsal"):
                        dorsal += 1
            if(dorsal >= palmar):
                prediction[unlabelled_img] = "dorsal"
            else:
                prediction[unlabelled_img] = "palmar"
          
            image_image_df = misc.load_from_pickle(reduced_pickle_file_folder,feature = (model + '_' + decomposition_model + "_image_image_df"))

        print(prediction)
        correct = 0
        class_list = unlabelled_metadata_df['imageName'].tolist()
        actual_class_list = unlabelled_metadata_df['aspectOfHand'].tolist()
        # print(actual_class_list)
        for image_name in prediction.keys():
            class_list = unlabelled_metadata_df['aspectOfHand'].where(unlabelled_metadata_df['imageName']==image_name)
                    
            class_list = [class_l for class_l in class_list if str(class_l) != 'nan']
            # print(str(class_list[0].split()[0]) + "--" + image_name )
            if(class_list[0].split()[0] == prediction.get(image_name)):
                correct += 1
        print(correct/len(prediction.keys()))
       
exit = 1    
while(exit!=0):
     
    if (classifier == '1'):
        svm()
        classifier = input("Choose the Classifier - \n. 1. SVM \n. 2. Decision Tree \n. 3. PPR Classifier \n. 4. Exit \n. ")
    elif (classifier == '2'):
        decision_tree_input()
        classifier = input("Choose the Classifier - \n. 1. SVM \n. 2. Decision Tree \n. 3. PPR Classifier \n. 4. Exit \n. ")
    elif (classifier == '3'):
        ppr_classifier()
        classifier = input("Choose the Classifier - \n. 1. SVM \n. 2. Decision Tree \n. 3. PPR Classifier \n. 4. Exit \n. ")
    elif (classifier == '4'):
        exit = 0
    else:
        print("Choose the correct classifier")
        classifier = input("Choose the Classifier - \n. 1. SVM \n. 2. Decision Tree \n. 3. PPR Classifier \n. 4. Exit \n. ")
        

     
