def metsumm(stepno=''): 
    import torch_xla.debug.metrics as met
    x = met.metrics_report().split('\n')
    for i, line in enumerate(x):
        if 'CompileTime' in line or 'aten::' in line:
            key = line.split()[-1]
            value = x[i+1].split()[-1]
            print('step {}, key {}, value {}'.format(stepno, key, value))
