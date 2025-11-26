MARP := marp
PREFIX := docs

OBJECTS += $(PREFIX)/index.html 
OBJECTS += $(PREFIX)/2025/lecture1.html 
OBJECTS += $(PREFIX)/2025/lecture2.html 
OBJECTS += $(PREFIX)/2025/lecture3.html 
OBJECTS += $(PREFIX)/2025/lecture4.html 
OBJECTS += $(PREFIX)/2025/lecture5.html 
OBJECTS += $(PREFIX)/2025/lecture6.html 
OBJECTS += $(PREFIX)/2025/lecture7.html 
OBJECTS += $(PREFIX)/2025/lecture8.html 
OBJECTS += $(PREFIX)/2025/lecture9.html 
OBJECTS += $(PREFIX)/2025/lecture10.html 
OBJECTS += $(PREFIX)/2025/lecture11.html 
OBJECTS += $(PREFIX)/2025/lecture12.html 
OBJECTS += $(PREFIX)/2025/lecture13.html 
# OBJECTS += $(PREFIX)/lecture1.html 
# OBJECTS += $(PREFIX)/lecture2.html 
# OBJECTS += $(PREFIX)/lecture3.html
# OBJECTS += $(PREFIX)/lecture4.html
# OBJECTS += $(PREFIX)/lecture5.html
# OBJECTS += $(PREFIX)/lecture6.html
# OBJECTS += $(PREFIX)/lecture7.html
# OBJECTS += $(PREFIX)/lecture8.html
# OBJECTS += $(PREFIX)/lecture9.html
# OBJECTS += $(PREFIX)/lecture10.html
# OBJECTS += $(PREFIX)/lecture11.html
# OBJECTS += $(PREFIX)/lecture12.html
# OBJECTS += $(PREFIX)/lecture13.html
# OBJECTS += $(PREFIX)/lecture14.html
# OBJECTS += $(PREFIX)/lecture_merge.html
# OBJECTS += $(PREFIX)/14.html
# OBJECTS += $(PREFIX)/15.html
# OBJECTS += $(PREFIX)/16.html

all: $(OBJECTS)

$(PREFIX)/%.html: %.md
	$(MARP) --html true $< -o $@ 
	cp -nr images/* $(PREFIX)/images/ &

.PHONY: clean

clean:
	rm -f $(PREFIX)/*.html
	rm -f $(PREFIX)/2025/*.html