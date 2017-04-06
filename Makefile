SUBDIRS := ass1-MPI

.PHONY : all $(SUBDIRS)
all : $(SUBDIRS)

$(SUBDIRS) :
	$(MAKE) -C $@ all
