from bifrost import map as BFMap 
def labelgroups(islanddata,ngrid,ints,nchan):
    # LABELGROUPS
    BFMap("""
auto counter=0;
for (auto i=0; i<%s; i++){
    for (auto j=0; j<%s; j++){
        if (isldata(time,chan,stokes,i,j) > 0) {
            counter = counter + 1;
            isldata(time,chan,stokes,i,j) = counter;
            }
        }
    }
            """ % (ngrid,ngrid),
        {'isldata':islanddata}, 
        axis_names=('time','chan','stokes'), shape=(ints,nchan,4))
        
def groupvertcols(idata,grid):
    BFMap("""
    if ((j>0) && (j<%i-1)) {
    	for (auto i = 1; i < %i-1; i++) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i - 1,j))) {
    			isldata(time,chan,stokes,i,j) = min(isldata(time,chan,stokes,i - 1,j), isldata(time,chan,stokes,i,j));
    		}
    	}
    	for (int i = %i - 2; i > 0; i--) {
    		if ((0 != isldata(time,chan,stokes,i,j)) && (0 != isldata(time,chan,stokes,i + 1,j))) {
    			isldata(time,chan,stokes,i,j) = min(isldata(time,chan,stokes,i + 1,j), isldata(time,chan,stokes,i,j));
    		}
    	}
     }
        """ % (grid,grid,grid,), 
        {'isldata':idata}, 
        axis_names=('time','chan','stokes','j'), shape=(idata.shape[0],idata.shape[1],idata.shape[2],idata.shape[3]))
        
def grouphorizrows(idata,grid):
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
        {'isldata':idata}, 
        axis_names=('time','chan','stokes','i'), shape=(idata.shape[0],idata.shape[1],idata.shape[2],idata.shape[3]))

def groupdiagcols(irdata,grid):
    BFMap("""
    for (auto j=2; j<%i-2; j++) {
    	for (auto i = 2; i < %i-2; i++) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i - 2,j))) {
    			rm(time,chan,stokes,i,j) = min(rm(time,chan,stokes,i - 2,j), rm(time,chan,stokes,i,j));
    		}
    	}
    	for (int i = %i - 3; i > 1; i--) {
    		if ((0 != rm(time,chan,stokes,i,j)) && (0 != rm(time,chan,stokes,i + 2,j))) {
    			rm(time,chan,stokes,i,j) = min(rm(time,chan,stokes,i + 2,j), rm(time,chan,stokes,i,j));
    		}
    	}
     }
        """ % (grid,grid,grid,), 
        {'rm':irdata}, 
        axis_names=('time','chan','stokes'), shape=(irdata.shape[0],irdata.shape[1],irdata.shape[2]))

def groupdiagrows(irdata, grid):
    BFMap("""
        for (auto i=2; i<%i-2; i++) {
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
        {'rm':irdata}, 
        axis_names=('time','chan','stokes'), shape=(irdata.shape[0],irdata.shape[1],irdata.shape[2]))

