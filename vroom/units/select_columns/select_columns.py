# the pragma directives tells imaginet to look for includes here
# you are not allowed to put comments in the include section
#pragma IMAGINET_INCLUDES_BEGIN
import numpy as np
#pragma IMAGINET_INCLUDES_END


#pragma IMAGINET_FRAGMENT_BEGIN "select_columns_py"
# the pragma directives makes this function accessible in the .imunit file using the name 'select_columns'

def select_cols_py(input, start_idx, end_idx, output):
    # Streams a 1D np.array for one given features at a time, therefore this slicing works
    result = input[start_idx:end_idx+1]
    np.copyto(output, result)

#pragma IMAGINET_FRAGMENT_END