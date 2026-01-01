"""
GUI MODULE 
This module makes main gui frame with Tkinter
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading

# import project modules
from data.data_handler import load_csv, prepare_data, get_column_info
from models.model_trainer import train_and_predict
from analysis.metrics import get_all_metrics, plot_confusion_matrix, generate_report

class MLApp:
    """
    Machine learning GUI APP main class
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning APP")
        self.root.geometry("850x750") 
        self.root.resizable(True, True)
        
        # data variables
        self.file_path = None
        self.dataframe = None
        
        # making interface
        self.create_widgets()
    
    def create_widgets(self):
        # main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== SECTION 1: Importing a Document =====
        file_frame = ttk.LabelFrame(main_frame, text="1. Import data", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No file selected!")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.file_button = ttk.Button(file_frame, text="Choose CSV file", command=self.select_file)
        self.file_button.pack(side=tk.RIGHT)
        
        # ===== SECTION 2: Choose Target Coulumn =====
        target_frame = ttk.LabelFrame(main_frame, text="2. Target coulumn", padding="10")
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(target_frame, text="Choose target coulumn:").pack(side=tk.LEFT, padx=(0, 10))
        self.target_combo = ttk.Combobox(target_frame, state="readonly", width=30)
        self.target_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ===== SECTION 3: Preprocessing Options =====
        preprocess_frame = ttk.LabelFrame(main_frame, text="3. preprocessing options", padding="10")
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.encoding_var = tk.BooleanVar(value=True)
        self.encoding_check = ttk.Checkbutton(
            preprocess_frame, 
            text="One-Hot Encoding (for categorical features)",
            variable=self.encoding_var
        )
        self.encoding_check.pack(anchor=tk.W)
        
        norm_frame = ttk.Frame(preprocess_frame)
        norm_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(norm_frame, text="Normalization:").pack(side=tk.LEFT, padx=(0, 10))
        self.norm_var = tk.StringVar(value="standard")
        
        ttk.Radiobutton(norm_frame, text="StandardScaler", variable=self.norm_var, value="standard").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(norm_frame, text="MinMaxScaler", variable=self.norm_var, value="minmax").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(norm_frame, text="None", variable=self.norm_var, value="none").pack(side=tk.LEFT, padx=5)
        
        # ===== BÖSECTIONLÜM 4: Choosing classification models =====
        model_frame = ttk.LabelFrame(main_frame, text="4. Choosing classification model", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="Choose a model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_var = tk.StringVar(value="Perceptron")
        self.model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.model_var,
            values=["Perceptron", "MLP", "Decision Tree"],
            state="readonly",
            width=15
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(model_frame, text="Hidden Layers (for MLP):").pack(side=tk.LEFT, padx=(0, 5))
        
        self.hidden_layers_var = tk.StringVar(value="100,50")
        self.hidden_layers_entry = ttk.Entry(
            model_frame, 
            textvariable=self.hidden_layers_var,
            width=15
        )
        self.hidden_layers_entry.pack(side=tk.LEFT)
        ttk.Label(model_frame, text="(EX: 100,50)").pack(side=tk.LEFT, padx=(5, 0))
        
        # ===== BÖLÜM 5: Train/Test Ratio =====
        split_frame = ttk.LabelFrame(main_frame, text="5. Train/Test Ratio", padding="10")
        split_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.split_label = ttk.Label(split_frame, text="Train: 70% | Test: 30%")
        self.split_label.pack(anchor=tk.W)
        
        self.split_var = tk.IntVar(value=70)
        self.split_slider = ttk.Scale(
            split_frame, from_=50, to=90, orient=tk.HORIZONTAL,
            variable=self.split_var, command=self.update_split_label
        )
        self.split_slider.pack(fill=tk.X, pady=(5, 0))

        
        # ===== BÖLÜM 6: Process Buttons =====
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Training one clasification model
        self.train_button = ttk.Button(
            button_frame, 
            text="🚀 train selected model",
            command=self.start_training
        )
        self.train_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=10)
        
        # Comparing button
        self.compare_button = ttk.Button(
            button_frame,
            text="⚡compare all models (table)",
            command=self.start_comparison
        )
        self.compare_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=10)
        
        # ===== BÖLÜM 7: Results =====
        result_frame = ttk.LabelFrame(main_frame, text="6. Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame, wrap=tk.WORD, font=("Consolas", 10), height=15
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        self.cm_button = ttk.Button(
            result_frame,
            text="📊 Show last models Confusion Matrix",
            command=self.show_confusion_matrix,
            state=tk.DISABLED
        )
        self.cm_button.pack(pady=(10, 0))
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        
        # to save results
        self.y_test = None
        self.y_pred = None
        self.label_encoder = None
    
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Choose CSV File",
            filetypes=[("CSV Files", "*.csv"), ("Tüm Dosyalar", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=file_path)
            try:
                self.dataframe = load_csv(file_path)
                columns = list(self.dataframe.columns)
                self.target_combo['values'] = columns
                self.target_combo.current(len(columns) - 1)
                messagebox.showinfo("Successful", f"File is imported!\nRow: {len(self.dataframe)}")
            except Exception as e:
                messagebox.showerror("Error", f"File couldn't read:\n{str(e)}")
    
    def update_split_label(self, value=None):
        train_ratio = self.split_var.get()
        self.split_label.config(text=f"Train: {train_ratio}% | Test: {100 - train_ratio}%")
    
    def get_hidden_layers(self):
        try:
            text = self.hidden_layers_var.get().strip()
            if not text:
                return (100, 50)
            
            layers = tuple(map(int, text.split(',')))
            return layers
        except ValueError:
            raise ValueError("Hidden layers must be comma separated integers (ex: 100,50)")

    def start_training(self):
        if not self.validate_inputs(): return
        
        try:
            hidden_layers = self.get_hidden_layers()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.toggle_buttons(False)
        
        self.result_text.delete(1.0, tk.END) 
        self.result_text.insert(tk.END, "Model training started...\n")
        
        self.progress.pack(fill=tk.X, pady=(10, 0))
        self.progress.start()
        
        threading.Thread(target=self.run_training, args=(hidden_layers,)).start()
    
    def start_comparison(self):
        """
        It runs all models in sequence and compares them.
        """
        if not self.validate_inputs(): return
        
        try:
            hidden_layers = self.get_hidden_layers()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.toggle_buttons(False)
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Running all models and comparing the results...\nPlease wait...\n")
        
        self.progress.pack(fill=tk.X, pady=(10, 0))
        self.progress.start()
        
        threading.Thread(target=self.run_comparison_process, args=(hidden_layers,)).start()

    def validate_inputs(self):
        if self.dataframe is None:
            messagebox.showwarning("Error", "Please select a CSV file first!")
            return False
        if not self.target_combo.get():
            messagebox.showwarning("Error", "Please select the target column!")
            return False
        return True

    def toggle_buttons(self, state):
        state_val = tk.NORMAL if state else tk.DISABLED
        self.train_button.config(state=state_val)
        self.compare_button.config(state=state_val)
        self.cm_button.config(state=tk.DISABLED) #disabled untill process is over

    def run_training(self, hidden_layers):
        try:
            target_column = self.target_combo.get()
            encoding = self.encoding_var.get()
            normalization = self.norm_var.get()
            model_name = self.model_var.get()
            train_ratio = self.split_var.get() / 100.0
            
            X_train, X_test, y_train, y_test, label_encoder = prepare_data(
                self.dataframe.copy(), target_column, encoding, normalization, train_ratio
            )
            
            model, y_pred = train_and_predict(model_name, X_train, X_test, y_train, hidden_layers=hidden_layers)
            report_text, metrics = generate_report(y_test, y_pred, model_name)
            
            self.y_test = y_test
            self.y_pred = y_pred
            self.label_encoder = label_encoder
            
            self.root.after(0, lambda: self.display_results(report_text))
            
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def run_comparison_process(self, hidden_layers):
        """
        The function that runs all models in the background.
        """
        try:
            target_column = self.target_combo.get()
            encoding = self.encoding_var.get()
            normalization = self.norm_var.get()
            train_ratio = self.split_var.get() / 100.0
            
            # Prepare the data once
            X_train, X_test, y_train, y_test, label_encoder = prepare_data(
                self.dataframe.copy(), target_column, encoding, normalization, train_ratio
            )
            
            models_to_test = ["Perceptron", "MLP", "Decision Tree"]
            results = []
            
            # To store the latest model data
            last_y_pred = None
            
            for model_name in models_to_test:
                # training the model
                model, y_pred = train_and_predict(model_name, X_train, X_test, y_train, hidden_layers=hidden_layers)
                # get the metrics
                metrics = get_all_metrics(y_test, y_pred)
                results.append((model_name, metrics))
                last_y_pred = y_pred
            
            # Assign the results to global variables
            self.y_test = y_test
            self.y_pred = last_y_pred
            self.label_encoder = label_encoder
            
            # Making table
            table_str = self.create_comparison_table(results)
            self.root.after(0, lambda: self.display_results(table_str))
            
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def create_comparison_table(self, results):
        """
        Making table from results
        """
        header = f"{'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n"
        separator = "-" * 65 + "\n"
        
        table = "\n" + "="*20 + " MODEL COMPARING TABLE " + "="*20 + "\n\n"
        table += header + separator
        
        for model_name, metrics in results:
            acc = metrics['Accuracy']
            prec = metrics['Precision']
            rec = metrics['Recall']
            f1 = metrics['F1-Score']
            
            row = f"{model_name:<15} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}\n"
            table += row

        table += separator
        
        return table
    
    def display_results(self, text):
        self.progress.stop()
        self.progress.pack_forget()
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        
        self.toggle_buttons(True)
        self.cm_button.config(state=tk.NORMAL)
    
    def display_error(self, message):
        self.progress.stop()
        self.progress.pack_forget()
        self.result_text.insert(tk.END, f"\n\nERROR:\n{message}")
        self.toggle_buttons(True)
        messagebox.showerror("Error", message)
    
    def show_confusion_matrix(self):
        if self.y_test is None or self.y_pred is None:
            messagebox.showwarning("Warning", "Train the model first!")
            return
        
        class_names = None
        if self.label_encoder is not None:
            class_names = list(self.label_encoder.classes_)
        plot_confusion_matrix(self.y_test, self.y_pred, class_names)

def run_app():
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()