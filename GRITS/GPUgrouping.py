from bifrost import map as BFMap 
def groupvertcols(data,grid):
    BFMap("""
    if ((j>0) && (j<%i-1)) {
    	for (auto i = 1; i < %i-1; i++) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i - 1,j))) {
    			isldata(time,chan,stokes,i,j) = min(isldata(time,chan,stokes,i - 1,j), isldata(time,chan,stokes,i,j));
    		}
    	}
    	for (int i = %i - 1; i > 0; i--) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i + 1,j))) {
    			isldata(time,chan,stokes,i,j) = min(isldata(time,chan,stokes,i + 1,j), isldata(time,chan,stokes,i,j));
    		}
    	}
     }
        """ % (grid,grid,grid,), 
        {'isldata':data}, 
        axis_names=('time','chan','stokes','j'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
        
def grouphorizrows(data,grid):
    BFMap("""
        if ((i>0) && (i<%i-1)) {
    	for (auto j = 1; j < %i-1; j++) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i,j + 1)) && (isldata(time,chan,stokes,i,j + 1) < isldata(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (isldata(time,chan,stokes,i,k) == isldata(time,chan,stokes,i,j)) isldata(time,chan,stokes,i,k) = isldata(time,chan,stokes,i,j + 1);
    		}
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i,j - 1)) && (isldata(time,chan,stokes,i,j - 1) < isldata(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (isldata(time,chan,stokes,i,k) == isldata(time,chan,stokes,i,j)) isldata(time,chan,stokes,i,k) = isldata(time,chan,stokes,i,j - 1);
    		}
    	}
    }
        """ % (grid,grid,grid,grid,), 
        {'isldata':data}, 
        axis_names=('time','chan','stokes','i'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))

def groupdiagcols(data,grid):
    BFMap("""
    if ((j>1) && (j<%i-2)) {
    	for (auto i = 2; i < %i-2; i++) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i - 2,j))) {
    			rm(time,chan,stokes,i,j) = min(rm(time,chan,stokes,i - 2,j), rm(time,chan,stokes,i,j));
    		}
    	}
    	for (int i = %i - 2; i > 1; i--) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i + 2,j))) {
    			rm(time,chan,stokes,i,j) = min(rm(time,chan,stokes,i + 2,j), rm(time,chan,stokes,i,j));
    		}
    	}
     }
        """ % (grid,grid,grid,), 
        {'rm':data}, 
        axis_names=('time','chan','stokes','j'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))

def groupdiagrows(data, grid):
    BFMap("""
        if ((i>1) && (i<%i-2)) {
    	for (auto j = 2; j < %i-2; j++) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i,j + 2)) && (rm(time,chan,stokes,i,j + 2) < rm(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (rm(time,chan,stokes,i,k) == rm(time,chan,stokes,i,j)) rm(time,chan,stokes,i,k) = rm(time,chan,stokes,i,j + 2);
    		}
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i,j - 2)) && (rm(time,chan,stokes,i,j - 2) < rm(time,chan,stokes,i,j))) {
    			for (auto k = 0; k < %i; k++) if (rm(time,chan,stokes,i,k) == rm(time,chan,stokes,i,j)) rm(time,chan,stokes,i,k) = rm(time,chan,stokes,i,j - 2);
    		}
    	}
    }
        """ % (grid,grid,grid,grid,), 
        {'rm':data}, 
        axis_names=('time','chan','stokes','i'), shape=(data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
