from pathlib import Path

from bs4 import BeautifulSoup
import re

class HTMLMetrics:

    def generate_html_metrics(self, html_filename):
        result = {}
        result["html_filename"] = str(html_filename)
        result["is_html_filename"] = Path(html_filename).is_file()
        if result["is_html_filename"]:    
            with open(html_filename, encoding="utf-8") as f:
                html = BeautifulSoup(f, "html.parser")            
            
            html_forms = html.find_all("form", attrs={"id": re.compile(".")})
            result["n_html_forms"] = html_forms is not None and len(html_forms) or 0
            
            html_buttons = html.find_all("button", attrs={"id": re.compile(".")})
            result["n_html_buttons"] = html_buttons is not None and len(html_buttons) or 0

            html_inputs = html.find_all("input", attrs={"id": re.compile(".")})
            result["n_html_inputs"] = html_inputs is not None and len(html_inputs) or 0

            html_input_buttons = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("button", flags=re.IGNORECASE)})
            result["n_html_input_button"] = html_input_buttons is not None and len(html_input_buttons) or 0

            html_input_submit = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("submit", flags=re.IGNORECASE)})
            result["n_html_input_submit"] = html_input_submit is not None and len(html_input_submit) or 0

            html_input_checkbox = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("checkbox", flags=re.IGNORECASE)})
            result["n_html_input_checkbox"] = html_input_checkbox is not None and len(html_input_checkbox) or 0

            html_input_date = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("date", flags=re.IGNORECASE)})
            result["n_html_input_date"] = html_input_date is not None and len(html_input_date) or 0

            html_input_datetime_local = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("datetime-local", flags=re.IGNORECASE)})
            result["n_html_input_datetime_local"] = html_input_datetime_local is not None and len(html_input_datetime_local) or 0

            html_input_email = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("email", flags=re.IGNORECASE)})
            result["n_html_input_email"] = html_input_email is not None and len(html_input_email) or 0

            html_input_password = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("password", flags=re.IGNORECASE)})
            result["n_html_input_password"] = html_input_password is not None and len(html_input_password) or 0

            html_input_number = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("number", flags=re.IGNORECASE)})
            result["n_html_input_number"] = html_input_number is not None and len(html_input_number) or 0

            html_input_reset = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("reset", flags=re.IGNORECASE)})
            result["n_html_input_reset"] = html_input_reset is not None and len(html_input_reset) or 0

            html_input_radio = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("radio", flags=re.IGNORECASE)})
            result["n_html_input_radio"] = html_input_radio is not None and len(html_input_radio) or 0

            html_input_range = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("range", flags=re.IGNORECASE)})
            result["n_html_input_range"] = html_input_range is not None and len(html_input_range) or 0

            html_input_text = html.find_all("input", attrs={"id": re.compile("."), "type":re.compile("text", flags=re.IGNORECASE)})
            result["n_html_input_text"] = html_input_text is not None and len(html_input_text) or 0

            html_labels = html.find_all("label", attrs={"for": re.compile(".")})
            result["n_html_label"] = html_labels is not None and len(html_labels) or 0

            html_selects = html.find_all("select", attrs={"id": re.compile(".")})
            result["n_html_select"] = html_selects is not None and len(html_selects) or 0

            html_options = html.find_all("option", attrs={"value": re.compile(".")})
            result["n_html_option"] = html_options is not None and len(html_options) or 0

            html_table = html.find_all("table")
            result["n_html_table"] = html_table is not None and len(html_table) or 0

        else:
            result["n_html_forms"] = 0
            result["n_html_buttons"] =  0
            result["n_html_inputs"] =  0
            result["n_html_input_button"] = 0
            result["n_html_input_submit"] = 0
            result["n_html_input_checkbox"] = 0
            result["n_html_input_date"] = 0
            result["n_html_input_datetime_local"] = 0
            result["n_html_input_email"] = 0
            result["n_html_input_password"] = 0
            result["n_html_input_number"] = 0
            result["n_html_input_reset"] = 0
            result["n_html_input_radio"] =  0
            result["n_html_input_range"] = 0
            result["n_html_input_text"] = 0
            result["n_html_label"] = 0
            result["n_html_select"] = 0
            result["n_html_option"] = 0        
            result["n_html_table"] = 0

        return result