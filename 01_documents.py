"""الملف ده بقرأ منه ملفات البي دي إف وبطلع كل صفحة لوحدها."""

# الملف ده فيه قراءة ملفات البي دي إف من مجلد المستندات.
# كل صفحة بتتحول لكائن مستقل ومعاها بيانات المصدر ورقم الصفحة.

# هنا بجيب أداة التعامل مع مسارات الملفات والمجلدات.
from pathlib import Path

# هنا بجيب أداة قراءة ملفات البي دي إف صفحة صفحة.
from langchain_community.document_loaders import PyPDFLoader


# هنا بحدد اسم المجلد اللي جواه ملفات البي دي إف.
DOCUMENTS_FOLDER = Path("documents")


# الفانكشن دي بتدور على ملفات البي دي إف، تقراها، وبترجع كل الصفحات في قائمة واحدة.
def load_documents():
    """بقرأ كل ملفات البي دي إف وبرجع الصفحات كلها في قائمة واحدة."""

    # هنا بعمل قائمة فاضية هجمع فيها كل الصفحات المقروءة.
    documents = []


    # هنا بشوف إن مجلد المستندات موجود قبل ما أحاول أقرأ منه.
    if not DOCUMENTS_FOLDER.exists():
        raise FileNotFoundError(
            f"Folder '{DOCUMENTS_FOLDER}' does not exist."
        )


    # هنا بجيب كل الملفات اللي امتدادها بي دي إف من المجلد.
    pdf_files = list(DOCUMENTS_FOLDER.glob("*.pdf"))

    # هنا بطلع رسالة واضحة وأوقف الشغل لو مفيش أي ملف بي دي إف.
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found inside '{DOCUMENTS_FOLDER}'."
        )


    # هنا بلف على الملفات واحد واحد وبقرأ كل صفحة.
    for file_path in pdf_files:
        print(f"Loading: {file_path}")

        # هنا بعمل قارئ البي دي إف للملف الحالي.
        loader = PyPDFLoader(str(file_path))

        # هنا بقرأ الملف، وكل صفحة بتطلع لوحدها.
        loaded_pages = loader.load()

        # هنا بضم الصفحات دي على باقي الصفحات.
        documents.extend(loaded_pages)

    # هنا برجع الصفحات علشان أكمل عليها باقي الخطوات.
    return documents


# الفانكشن دي بتشغل اختبار الملف لما أشغله مباشرة من سطر الأوامر.
def main():
    try:
        docs = load_documents()

        print("=" * 50)
        print(f"Loaded PDF Pages: {len(docs)}")
        print("=" * 50)

        for doc in docs[:5]:
            print(
                f"Source: {doc.metadata.get('source', 'Unknown')}"
            )
            print(
                f"Page: {doc.metadata.get('page', 'Unknown')}"
            )
            print("-" * 50)

    except Exception as error:
        print("=" * 50)
        print("ERROR")
        print("=" * 50)
        print(error)


if __name__ == "__main__":
    main()
