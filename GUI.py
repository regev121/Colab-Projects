import customtkinter as ctk
from CTkMessagebox import CTkMessagebox

from Predict import PREDICT
def project():
    review_text = review_entry1.get()
    genres_text = review_entry2.get()
    if (PREDICT(review_text, genres_text) == True):
        CTkMessagebox(title="Info", message='Predicted Game Will Succeed')
    else:
        CTkMessagebox(title="Info", message='Predicted Game Will Not Succeed')


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

root = ctk.CTk()
root.geometry("500x350")

frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)


review_label1 = ctk.CTkLabel(master=frame, text="Enter the review of the game you want to predict it's success or failure:")
review_label1.pack(pady=12, padx=10)
review_entry1 = ctk.CTkEntry(master=frame, width=500, placeholder_text="• Enter your review here.")
review_entry1.pack(pady=12, padx=10)

review_label2 = ctk.CTkLabel(master=frame, text="enter the genres of the game you want to predict\n the genres that can be used in this model are Action, Adventure, Casual, Early Access, Free to Play, Indie, Massively Multiplayer, Nudity, RPG, Racing, Simulation, Sports, and Strategy.")
review_label2.pack(pady=12, padx=10)
review_entry2 = ctk.CTkEntry(master=frame, width=500, placeholder_text="• Enter your genres here.")
review_entry2.pack(pady=12, padx=10)

predict_button = ctk.CTkButton(master=frame, text="Predict", command=project)
predict_button.pack(pady=12, padx=10)


root.mainloop()
