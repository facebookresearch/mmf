
# Adopted from
# https://github.com/pytorch/fairseq/blob/master/fairseq/distributed_utils.py

import pickle
import torch

MAX_SIZE_LIMIT = 65533
BYTE_SIZE = 256


def enc_obj2bytes(obj, max_size=4094):
    """
    Encode Python objects to PyTorch byte tensors
    """
    assert max_size <= MAX_SIZE_LIMIT
    byte_tensor = torch.zeros(max_size, dtype=torch.uint8)

    obj_enc = pickle.dumps(obj)
    obj_size = len(obj_enc)
    if obj_size > max_size:
        raise Exception(
            'objects too large: object size {}, max size {}'.format(
                obj_size, max_size
            )
        )

    byte_tensor[0] = obj_size // 256
    byte_tensor[1] = obj_size % 256
    byte_tensor[2:2+obj_size] = torch.ByteTensor(list(obj_enc))
    return byte_tensor


def dec_bytes2obj(byte_tensor, max_size=4094):
    """
    Decode PyTorch byte tensors to Python objects
    """
    assert max_size <= MAX_SIZE_LIMIT

    obj_size = byte_tensor[0].item() * 256 + byte_tensor[1].item()
    obj_enc = bytes(byte_tensor[2:2+obj_size].tolist())
    obj = pickle.loads(obj_enc)
    return obj


if __name__ == '__main__':
    test_obj = [1, '2', {3: 4}, [5]]
    test_obj_bytes = enc_obj2bytes(test_obj)
    test_obj_dec = dec_bytes2obj(test_obj_bytes)
    print(test_obj_dec == test_obj)
