# coding=utf-8
import sys
import os
from huawei_test.official import predictor_tuihuo

ecsDataPath = 'C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\official\case\TrainData_2015.1.1_2015.5.24.txt'
inputFilePath = 'C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\official\case\input_6flavors_cpu_7days.txt'
resultFilePath = 'C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\official\output1.txt'
testDataPath = 'C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\official\case\TestData_2015.5.25_20155.31.txt'


def main():
    # print ('main function begin.')
    # if len(sys.argv) != 4:
    #     print ('parameter is incorrect!')
    #     print ('Usage: python esc.py ecsDataPath inputFilePath resultFilePath')
    #     exit(1)
    # # Read the input files
    # ecsDataPath = sys.argv[1]
    # inputFilePath = sys.argv[2]
    # resultFilePath = sys.argv[3]

    ecs_infor_array = read_lines(ecsDataPath)
    input_file_array = read_lines(inputFilePath)
    test_file_array = read_lines(testDataPath)

    # implementate the function predictVm
    predic_result = predictor_tuihuo.predict_vm(ecs_infor_array, input_file_array)
    # write the result to output file
    if len(predic_result) != 0:
        write_result(predic_result, resultFilePath)
    else:
        predic_result.append("NA")
        write_result(predic_result, resultFilePath)
    print('main function end.')


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                line = line.rstrip()
                if line != '':
                    array.append(line)
        return array
    else:
        print('file not exist: ' + file_path)
        return None


if __name__ == "__main__":
    main()
