EXECUTABLE = combine_gram_count
LIBRARY =
OBJECT =
TEST =
CXX ?= c++
CXXFLAGS += -std=c++17 -O3 -I.
LDFLAGS += -lcppglob -lstdc++fs -lgflags -lglog
AR ?= ar
ARFLAGS +=

TEST_LDFLAGS += $(LDFLAGS) -lgtest -lgtest_main

all : $(EXECUTABLE) $(LIBRARY) $(OBJECT) $(TEST)

clean :
	-rm $(EXECUTABLE) $(LIBRARY) $(OBJECT) $(TEST)

executable : $(EXECUTABLE)

library: $(LIBRARY)

object: $(OBJECT)

test : $(TEST)

COMBINE_GRAM_COUNT_SOURCE = combine_gram_count.cpp
COMBINE_GRAM_COUNT_CXXFLAGS += $(CXXFLAGS)
COMBINE_GRAM_COUNT_LDFLAGS += $(LDFLAGS)
combine_gram_count : $(COMBINE_GRAM_COUNT_SOURCE)
	$(CXX) -o $@ $(COMBINE_GRAM_COUNT_CXXFLAGS) \
	$(COMBINE_GRAM_COUNT_SOURCE) $(COMBINE_GRAM_COUNT_LDFLAGS)
