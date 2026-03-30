function previewImage() {
    const file = document.getElementById("file").files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const img = document.getElementById("preview");
        img.src = e.target.result;
        img.style.display = "block";
    }

    if (file) {
        reader.readAsDataURL(file);
    }
}

function showLoader() {
    document.getElementById("loader").style.display = "block";
}
function toggleForm() {
    const login = document.getElementById("loginForm");
    const signup = document.getElementById("signupForm");
    const title = document.getElementById("form-title");

    if (signup.classList.contains("hidden")) {
        signup.classList.remove("hidden");
        login.classList.add("hidden");
        title.innerText = "Sign Up";
    } else {
        signup.classList.add("hidden");
        login.classList.remove("hidden");
        title.innerText = "Login";
    }
}