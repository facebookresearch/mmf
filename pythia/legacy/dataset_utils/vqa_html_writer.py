# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


html_header = """
<html>
<title>W3.CSS Template</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body,h1,h2,h3,h4,h5,h6 {font-family: "Karma", sans-serif}
.w3-bar-block .w3-bar-item {padding:20px}
</style>
<body>

<div class="w3-main w3-content w3-padding"
style="max-width:1200px;margin-top:100px">

"""

html_footer = """
</body>
</html>
"""

row_header = """
<div class="w3-row-padding w3-padding-16 w3-center" id="vqa">
"""

element_header = """
<div class="w3-quarter">
"""


class vqa_html_writer:
    def __init__(self, file_path, elements_per_row=4):
        self._writer = open(file_path, "w")
        self._writer.write(html_header)
        self.count = 0
        self.elements_per_row = elements_per_row

    def write_element(self, image, **kwarg):
        if self.count % self.elements_per_row == 0:
            self._writer.write(row_header + "\n")
        self._writer.write(element_header)
        self._writer.write('<img src=" ' + image + '" width = 100%">')
        for key, value in kwarg.items():
            self._writer.write("<p>%s : %s</p>" % (key, value))
        self._writer.write("</div>")
        self.count += 1
        if self.count % self.elements_per_row == 0 and self.count > 0:
            self._writer.write("</div>")

    def close(self):
        if self.count % self.elements_per_row != 0:
            self._writer.write("</div>")
        self._writer.write(html_footer)
        self._writer.close()


if __name__ == "__main__":
    html_writer = vqa_html_writer("/Users/tinayujiang/temp/test.html", 4)
    n = 10
    for i in range(10):
        image_path = (
            "/Users/tinayujiang/work/VQA/data_analysis/val2014/"
            + "COCO_val2014_000000290951.jpg"
        )
        info = {"question": "abcfs efc?", "answers": " wdds cdsde"}
        html_writer.write_element(image_path, **info)

    html_writer.close()
