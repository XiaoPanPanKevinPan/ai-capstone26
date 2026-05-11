import json
import os
import numpy as np
from fk import your_fk, get_ur5_DH_params
from ik import your_ik

# Error thresholds from the original assignment files
FK_ERROR_THRESH = 0.005
JACOBIAN_ERROR_THRESH = 0.05
IK_ERROR_THRESH = 0.02

def verify_fk():
    print("============================ Task 1 : Forward Kinematic (CPU Verification) ============================\n")
    dh_params = get_ur5_DH_params()
    base_pos = np.asarray([-0.2, 0.13, 0.6], dtype=np.float64)
    
    testcase_files = [
        'test_case/fk_test_case_easy.json',
        'test_case/fk_test_case_medium.json',
        'test_case/fk_test_case_hard.json',
    ]
    
    for testcase_file in testcase_files:
        if not os.path.exists(testcase_file):
            print(f"[Warning] Test case file {testcase_file} not found. Skipping...")
            continue
            
        with open(testcase_file, 'r') as f_in:
            fk_dict = json.load(f_in)

        test_case_name = os.path.split(testcase_file)[-1]
        joint_poses = fk_dict['joint_poses']
        poses = fk_dict['poses']
        jacobians = fk_dict['jacobian']
        cases_num = len(joint_poses)

        fk_error_cnt = 0
        jacobian_error_cnt = 0

        for i in range(cases_num):
            your_pose, your_jacobian = your_fk(dh_params, joint_poses[i], base_pos)
            gt_pose = np.asarray(poses[i])
            gt_jac = np.asarray(jacobians[i])

            fk_error = np.linalg.norm(your_pose - gt_pose, ord=2)
            if fk_error > FK_ERROR_THRESH:
                fk_error_cnt += 1

            jacobian_error = np.linalg.norm(your_jacobian - gt_jac, ord=2)
            if jacobian_error > JACOBIAN_ERROR_THRESH:
                jacobian_error_cnt += 1

        print(f"- Testcase file : {test_case_name}")
        print(f"  FK Error Count : {fk_error_cnt:4d} / {cases_num:4d}")
        print(f"  Jacobian Error Count : {jacobian_error_cnt:4d} / {cases_num:4d}")
        print("-" * 50)

def verify_ik():
    print("\n============================ Task 2 : Inverse Kinematic (CPU Verification) ============================\n")
    dh_params = get_ur5_DH_params()
    base_pos = np.asarray([-0.2, 0.13, 0.6], dtype=np.float64)
    
    testcase_files = [
        'test_case/ik_test_case_easy.json',
        'test_case/ik_test_case_medium.json',
        'test_case/ik_test_case_hard.json',
    ]
    
    # Standard initial joint state from the assignment
    q_init = np.asarray([
        -3.141592642791131,
        -1.5707963240621052,
        1.5707963521600738,
        -1.5707963267948966,
        -1.5707963267948966,
        1.06243199169874e-08,
    ], dtype=np.float64)

    for testcase_file in testcase_files:
        if not os.path.exists(testcase_file):
            print(f"[Warning] Test case file {testcase_file} not found. Skipping...")
            continue
            
        with open(testcase_file, 'r') as f_in:
            ik_dict = json.load(f_in)

        test_case_name = os.path.split(testcase_file)[-1]
        target_poses = ik_dict['next_poses']
        cases_num = len(target_poses)
        
        ik_error_cnt = 0
        q_curr = q_init.copy()

        for target_p in target_poses:
            q_sol = your_ik(target_p, base_pos, q_init=q_curr)
            q_curr = np.asarray(q_sol, dtype=np.float64)
            
            # Verify IK solution by calculating FK and checking pose error
            solved_pose, _ = your_fk(dh_params, q_curr, base_pos)
            ik_error = np.linalg.norm(solved_pose - np.asarray(target_p), ord=2)
            
            if ik_error > IK_ERROR_THRESH:
                ik_error_cnt += 1

        print(f"- Testcase file : {test_case_name}")
        print(f"  IK Error Count : {ik_error_cnt:4d} / {cases_num:4d}")
        print("-" * 50)

if __name__ == "__main__":
    verify_fk()
    verify_ik()
