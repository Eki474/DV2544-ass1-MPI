SUBDIRS := ass1-MPI lab2-pthread ass2-pthread

.PHONY : all $(SUBDIRS)
all : $(SUBDIRS)

$(SUBDIRS) :
	$(MAKE) -C $@ all
